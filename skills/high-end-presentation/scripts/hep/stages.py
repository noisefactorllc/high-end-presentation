"""Pipeline stages. Each takes a Job and returns a JSON-able summary."""
import json
from pathlib import Path

import cv2
import numpy as np

from . import analyze as analyze_mod
from . import color, endpoints, fal, fidelity, grade, lens, locate, media, prompts, timelapse
from .job import Job, JobError

GATE_MIN_CORR = 0.6
GATE_MIN_TILE = 0.35
FINAL_MIN_CORR = 0.85
ECHO_STRENGTH = 0.15


GLAZED_SCENES = {"storefront", "glass-display", "screens"}
MAX_PIECES = 6
MAX_QUAD_OVERLAP = 0.05


class GateFailed(RuntimeError):
    pass


def _client():
    return fal.FalClient(fal.key_from_env())


def init(root, pieces, prompt, *, scene="freeform", energy="calm", aspect="4:5", resolution="2K",
         duration=10, echo=False, frame=None, name=None, image_endpoint=None, video_endpoint=None):
    pieces = [pieces] if isinstance(pieces, (str, Path)) else list(pieces)
    if not 1 <= len(pieces) <= MAX_PIECES:
        raise JobError(f"give between 1 and {MAX_PIECES} pieces")
    pieces = [Path(p).resolve() for p in pieces]
    for p in pieces:
        if not p.exists():
            raise JobError(f"piece not found: {p}")
    if energy not in timelapse.ENERGY:
        raise JobError(f"energy must be one of {', '.join(timelapse.ENERGY)}")
    if aspect not in endpoints.ASPECTS:
        raise JobError(f"aspect must be one of {', '.join(sorted(endpoints.ASPECTS))}")
    if resolution not in endpoints.RESOLUTIONS:
        raise JobError(f"resolution must be one of {', '.join(sorted(endpoints.RESOLUTIONS))}")
    job = Job.create(root, {
        "pieces": [str(p) for p in pieces], "prompt": prompt, "scene": scene, "energy": energy,
        "aspect": aspect, "resolution": resolution, "duration": int(duration), "echo": bool(echo),
        "frame": frame, "name": name or pieces[0].stem,
        "glazed": scene in GLAZED_SCENES,
        "endpoints": {"image": image_endpoint or endpoints.IMAGE_DEFAULT,
                      "video": video_endpoint or endpoints.VIDEO_DEFAULT,
                      "depth": endpoints.DEPTH_DEFAULT},
    })
    job.done("init")
    return {"job": str(job.root), "pieces": len(pieces), "next": "analyze"}


def _merged_palette(palettes, k=8):
    merged = [dict(c, weight=c["weight"] / len(palettes)) for pal in palettes for c in pal]
    return sorted(merged, key=lambda c: -c["weight"])[:k]


def analyze(job):
    job.require("init")
    results, previews = [], []
    for i, src in enumerate(job["pieces"]):
        result, still = analyze_mod.analyze_piece(src)
        if job.get("frame") is not None and media.is_video(src):
            still = media.read_frame(src, int(job["frame"]))
            result["source"]["frame_index"] = int(job["frame"])
        media.save_image(job.file(f"piece-{i}.png"), still)
        previews.append(str(media.save_image(job.file(f"piece-{i}-preview.jpg"), still)))
        if job.get("echo") and media.is_video(src):
            frames = [f for _, f in media.iter_frames(src, step=max(1, result["source"]["frames"] // 48))]
            echo = color.linear_to_srgb(np.mean([color.srgb_to_linear(f) for f in frames], 0))
            media.save_image(job.file(f"piece-{i}-echo.png"), echo)
        results.append(result)
    job.data["analysis"] = {"pieces": results, "palette": _merged_palette([r["palette"] for r in results])}
    job.done("analyze")
    return {"analysis": job.data["analysis"], "piece_previews": previews,
            "next": "look at the piece previews, then write the brief (stage `brief`)"}


BRIEF_KEYS = {"themes", "mood", "scene", "motion"}


def brief(job, data):
    job.require("analyze")
    missing = BRIEF_KEYS - set(data)
    if missing:
        raise JobError(f"brief is missing: {', '.join(sorted(missing))}")
    job.data["brief"] = data
    if "glazed" in data:
        job.data["glazed"] = bool(data["glazed"])
    job.done("brief")
    return {"brief": data, "next": "still (or draft after still)"}


def _presented_pieces(job):
    """Each piece as it must appear: its still, or the still with a faint echo."""
    out = []
    for i in range(len(job["pieces"])):
        piece = media.load_image_any_depth(job.file(f"piece-{i}.png"))
        echo_path = job.root / f"piece-{i}-echo.png"
        if job.get("echo") and echo_path.exists():
            echo = media.load_image_any_depth(echo_path)
            lin = (1 - ECHO_STRENGTH) * color.srgb_to_linear(piece) + ECHO_STRENGTH * color.srgb_to_linear(echo)
            piece = color.linear_to_srgb(lin)
        out.append(piece)
    return out


def _overlap(quads, shape):
    """Largest pairwise overlap between located quads, as a share of the smaller."""
    h, w = shape[:2]
    s = min(1.0, 512 / max(h, w))
    masks = []
    for q in quads:
        m = np.zeros((round(h * s), round(w * s)), np.uint8)
        cv2.fillConvexPoly(m, np.round(np.asarray(q) * s).astype(np.int32), 1)
        masks.append(m.astype(bool))
    worst = 0.0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            inter = (masks[i] & masks[j]).sum()
            worst = max(worst, inter / max(1, min(masks[i].sum(), masks[j].sum())))
    return float(worst)


def still(job, attempts=3, timeout=600):
    job.require("brief")
    pieces = _presented_pieces(job)
    prompt = prompts.still_prompt(job["brief"]["scene"], [(p.shape[1], p.shape[0]) for p in pieces])
    ep = job["endpoints"]["image"]
    client = _client()
    uris = [fal.data_uri(p, max_side=2048) for p in pieces]
    history = job.data.setdefault("still_attempts", [])
    for n in range(len(history), len(history) + attempts):
        args = endpoints.image_args(ep, prompt, uris, job["aspect"], job["resolution"], seed=1000 + n)
        res = client.run(ep, args, job.data["fal"], f"still-{n}", timeout, job.save)
        path = fal.download(endpoints.image_url(res), job.file("still", f"attempt-{n}.png"))
        scene = media.load_image(path)
        gates = [fidelity.gate(p, scene, min_corr=GATE_MIN_CORR, min_tile=GATE_MIN_TILE) for p in pieces]
        failed = [(i, g) for i, g in enumerate(gates) if not g.passed]
        reason = "; ".join(f"piece {i}: {g.reason}" for i, g in failed)
        overlap = _overlap([g.located.quad for g in gates], scene.shape) if not failed and len(gates) > 1 else 0.0
        if not failed and overlap > MAX_QUAD_OVERLAP:
            reason = f"pieces-overlap:{overlap:.2f}"
        history.append({"attempt": n, "file": str(path), "passed": not reason, "reason": reason,
                        "pieces": [g.to_json() for g in gates]})
        job.save()
        if not reason:
            plate, mask = scene, np.zeros(scene.shape[:2], np.float32)
            for p, g in zip(pieces, gates):
                plate, m, _ = fidelity.repair(p, plate, g.located, glazed=job.get("glazed", False))
                mask = np.maximum(mask, m)
            media.save_image(job.file("still", "plate.png"), plate)
            media.save_image(job.file("still", "plate-preview.jpg"), plate)
            np.save(job.file("still", "mask.npy"), mask.astype(np.float16))
            job.data["located"] = [g.located.to_json() for g in gates]
            job.data["plate_size"] = [plate.shape[1], plate.shape[0]]
            summary = [{"corr": round(g.corr, 4), "worst_tile": round(g.worst, 4),
                        "aspect_error": round(g.aspect_error, 4)} for g in gates]
            job.done("still", attempt=n, pieces=summary)
            return {"passed": True, "attempt": n, "pieces": summary,
                    "plate_preview": str(job.file("still", "plate-preview.jpg")), "reference": str(path),
                    "next": "look at plate-preview.jpg; then `draft` for a preview or `video`"}
    raise GateFailed(json.dumps({"passed": False, "attempts": [
        {"attempt": a["attempt"], "file": a["file"], "reason": a["reason"]} for a in history[-attempts:]]}))


def _plate(job):
    plate = media.load_image_any_depth(job.file("still", "plate.png"))
    mask = np.load(job.file("still", "mask.npy")).astype(np.float32)
    return plate, mask


def _depth(job, plate, timeout=300):
    p = job.root / "depth.png"
    if not p.exists():
        ep = job["endpoints"]["depth"]
        res = _client().run(ep, endpoints.depth_args(ep, fal.data_uri(plate, max_side=2048)),
                            job.data["fal"], "depth", timeout, job.save)
        fal.download(endpoints.depth_url(res), p)
    d = media.load_image(p).mean(-1)
    return cv2.resize(d, (plate.shape[1], plate.shape[0]), interpolation=cv2.INTER_LINEAR)


def _look(job, img_lin, plate, mask, opts):
    depth = _depth(job, plate)
    img_lin, coc = lens.depth_of_field(img_lin, depth, mask, opts.get("focus", 1.0))
    img_lin = lens.vignette(img_lin, opts.get("vignette", 0.22))
    return grade.apply(img_lin, job["analysis"]["palette"], mask, tone=opts.get("tone", 0.15),
                       contrast=opts.get("contrast", 0.08), grain_amount=opts.get("grain", 0.012),
                       piece_strength=opts.get("piece_grade", 0.5))


def _look_opts(job):
    b = job.get("brief", {})
    return {k: b[k] for k in ("focus", "vignette", "tone", "contrast", "grain", "piece_grade") if k in b}


def draft(job):
    job.require("still")
    plate, mask = _plate(job)
    out = _look(job, color.srgb_to_linear(plate), plate, mask, _look_opts(job))
    path = media.save_image(job.file("out", f"{job['name']}-draft.jpg"), out)
    return {"draft": str(path), "next": "video (the paid render) or adjust the brief and rerun still"}


def video(job, timeout=900):
    job.require("still")
    plate, _ = _plate(job)
    ep = job["endpoints"]["video"]
    prompt = prompts.video_prompt(job["brief"].get("motion", ""), job["energy"])
    args = endpoints.video_args(ep, prompt, prompts.VIDEO_NEGATIVE, fal.data_uri(plate, max_side=2048),
                                job["duration"])
    res = _client().run(ep, args, job.data["fal"], "video", timeout, job.save)
    path = fal.download(endpoints.video_url(res), job.file("video", "clip.mp4"))
    info = media.video_info(path)
    job.done("video", **info)
    return {"video": str(path), **info, "next": "develop"}


def develop(job):
    job.require("video")
    plate, mask = _plate(job)
    st = timelapse.stack(job.file("video", "clip.mp4"), plate)
    out, report = timelapse.develop(plate, st, mask, job["energy"], glazed=job.get("glazed", False))
    np.save(job.file("develop", "activity.npy"), report.pop("activity"))
    np.save(job.file("develop", "exposure.npy"), out.astype(np.float16))
    media.save_image(job.file("develop", "exposure-preview.jpg"), np.clip(color.linear_to_srgb(out), 0, 1))
    job.data["develop"] = report
    job.done("develop")
    return {"report": report, "preview": str(job.file("develop", "exposure-preview.jpg")), "next": "finish"}


DEVELOP_WARNINGS = ("camera-moved", "no-motion", "background:median")


def _warnings(job):
    notes = job.get("develop", {}).get("notes", [])
    return list(job["warnings"]) + [n for n in notes if n.startswith(DEVELOP_WARNINGS)]


def finish(job):
    job.require("develop")
    plate, mask = _plate(job)
    pieces = _presented_pieces(job)
    exposure = np.load(job.file("develop", "exposure.npy")).astype(np.float32)
    out = _look(job, exposure, plate, mask, _look_opts(job))
    activity = np.load(job.file("develop", "activity.npy")).astype(np.float32)
    scores = [fidelity.verify(p, out, locate.Located.from_json(l), exclude=activity)
              for p, l in zip(pieces, job["located"])]
    if min(scores) < FINAL_MIN_CORR:
        raise GateFailed(json.dumps({"final_verify": [round(x, 4) for x in scores], "min": FINAL_MIN_CORR,
                                     "message": "a finished piece region does not match its original"}))
    png = media.save_image(job.file("out", f"{job['name']}-presentation.png"), out)
    jpg = media.save_image(job.file("out", f"{job['name']}-presentation.jpg"), out)
    manifest = {
        "pieces": job["pieces"], "prompt": job["prompt"], "scene": job["scene"], "energy": job["energy"],
        "aspect": job["aspect"], "resolution": job["resolution"], "size": [out.shape[1], out.shape[0]],
        "brief": job["brief"], "endpoints": job["endpoints"],
        "requests": {k: v.get("request_id") for k, v in job["fal"].items()},
        "gate": job["stages"]["still"], "final_verify": [round(x, 4) for x in scores],
        "develop": job["develop"], "warnings": _warnings(job), "outputs": [str(png), str(jpg)],
    }
    job.file("out", "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    job.done("finish", final_verify=scores)
    return {"presentation": str(png), "jpeg": str(jpg), "final_verify": [round(x, 4) for x in scores],
            "warnings": _warnings(job)}


def status(job):
    return {"job": str(job.root), "stages": {s: job.is_done(s) for s in
            ["init", "analyze", "brief", "still", "video", "develop", "finish"]},
            "warnings": _warnings(job), "pending": {k: v["request_id"] for k, v in job["fal"].items()
                                                     if not v.get("done")}}
