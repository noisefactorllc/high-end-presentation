"""Pipeline stages. Each takes a Job and returns a JSON-able summary."""
import json
import shutil
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


class GateFailed(RuntimeError):
    pass


def _client():
    return fal.FalClient(fal.key_from_env())


def init(root, piece, prompt, *, scene="freeform", energy="calm", aspect="4:5", resolution="2K",
         duration=10, echo=False, frame=None, name=None, image_endpoint=None, video_endpoint=None):
    piece = Path(piece).resolve()
    if not piece.exists():
        raise JobError(f"piece not found: {piece}")
    if energy not in timelapse.ENERGY:
        raise JobError(f"energy must be one of {', '.join(timelapse.ENERGY)}")
    if aspect not in endpoints.ASPECTS:
        raise JobError(f"aspect must be one of {', '.join(sorted(endpoints.ASPECTS))}")
    if resolution not in endpoints.RESOLUTIONS:
        raise JobError(f"resolution must be one of {', '.join(sorted(endpoints.RESOLUTIONS))}")
    job = Job.create(root, {
        "piece": str(piece), "prompt": prompt, "scene": scene, "energy": energy, "aspect": aspect,
        "resolution": resolution, "duration": int(duration), "echo": bool(echo), "frame": frame,
        "name": name or piece.stem,
        "endpoints": {"image": image_endpoint or endpoints.IMAGE_DEFAULT,
                      "video": video_endpoint or endpoints.VIDEO_DEFAULT,
                      "depth": endpoints.DEPTH_DEFAULT},
    })
    job.done("init")
    return {"job": str(job.root), "next": "analyze"}


def analyze(job):
    job.require("init")
    src = job["piece"]
    result, still = analyze_mod.analyze_piece(src)
    if job.get("frame") is not None and media.is_video(src):
        still = media.read_frame(src, int(job["frame"]))
        result["source"]["frame_index"] = int(job["frame"])
    media.save_image(job.file("piece.png"), still)
    if job.get("echo") and media.is_video(src):
        frames = [f for _, f in media.iter_frames(src, step=max(1, result["source"]["frames"] // 48))]
        echo = color.linear_to_srgb(np.mean([color.srgb_to_linear(f) for f in frames], 0))
        media.save_image(job.file("piece-echo.png"), echo)
    job.data["analysis"] = result
    job.done("analyze")
    return {"analysis": result, "piece_still": str(job.file("piece.png")),
            "next": "look at piece.png, then write the brief (stage `brief`)"}


BRIEF_KEYS = {"themes", "mood", "scene", "motion"}


def brief(job, data):
    job.require("analyze")
    missing = BRIEF_KEYS - set(data)
    if missing:
        raise JobError(f"brief is missing: {', '.join(sorted(missing))}")
    job.data["brief"] = data
    job.done("brief")
    return {"brief": data, "next": "still (or draft after still)"}


def _presented_piece(job):
    """The piece as it must appear: the still, or the still with a faint echo."""
    piece = media.load_image_any_depth(job.file("piece.png"))
    echo_path = job.root / "piece-echo.png"
    if job.get("echo") and echo_path.exists():
        echo = media.load_image_any_depth(echo_path)
        lin = (1 - ECHO_STRENGTH) * color.srgb_to_linear(piece) + ECHO_STRENGTH * color.srgb_to_linear(echo)
        piece = color.linear_to_srgb(lin)
    return piece


def still(job, attempts=3, timeout=600):
    job.require("brief")
    piece = _presented_piece(job)
    ph, pw = piece.shape[:2]
    prompt = prompts.still_prompt(job["brief"]["scene"], pw, ph)
    ep = job["endpoints"]["image"]
    client = _client()
    piece_uri = fal.data_uri(piece, max_side=2048)
    history = job.data.setdefault("still_attempts", [])
    for n in range(len(history), len(history) + attempts):
        args = endpoints.image_args(ep, prompt, piece_uri, job["aspect"], job["resolution"], seed=1000 + n)
        res = client.run(ep, args, job.data["fal"], f"still-{n}", timeout, job.save)
        path = fal.download(endpoints.image_url(res), job.file("still", f"attempt-{n}.png"))
        scene = media.load_image(path)
        g = fidelity.gate(piece, scene, min_corr=GATE_MIN_CORR, min_tile=GATE_MIN_TILE)
        history.append({"attempt": n, "file": str(path), **g.to_json()})
        job.save()
        if g.passed:
            plate, mask, _ = fidelity.repair(piece, scene, g.located)
            media.save_image(job.file("still", "plate.png"), plate)
            np.save(job.file("still", "mask.npy"), mask.astype(np.float16))
            job.data["located"] = g.located.to_json()
            job.data["plate_size"] = [plate.shape[1], plate.shape[0]]
            job.done("still", attempt=n, corr=g.corr, worst_tile=g.worst)
            return {"passed": True, "attempt": n, "gate": g.to_json() | {"located": None},
                    "plate": str(job.file("still", "plate.png")), "reference": str(path),
                    "next": "look at the plate; then `draft` for a preview or `video`"}
    raise GateFailed(json.dumps({"passed": False, "attempts": [
        {k: a[k] for k in ("attempt", "file", "corr", "worst_tile", "reason")} for a in history[-attempts:]]}))


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
    out, report = timelapse.develop(plate, st, mask, job["energy"])
    np.save(job.file("develop", "exposure.npy"), out.astype(np.float16))
    media.save_image(job.file("develop", "exposure-preview.jpg"), np.clip(color.linear_to_srgb(out), 0, 1))
    job.data["develop"] = report
    job.done("develop")
    return {"report": report, "preview": str(job.file("develop", "exposure-preview.jpg")), "next": "finish"}


DEVELOP_WARNINGS = ("camera-moved", "no-motion", "background:median", "occlusion-removed")


def _warnings(job):
    notes = job.get("develop", {}).get("notes", [])
    return list(job["warnings"]) + [n for n in notes if n.startswith(DEVELOP_WARNINGS)]


def finish(job):
    job.require("develop")
    plate, mask = _plate(job)
    piece = _presented_piece(job)
    exposure = np.load(job.file("develop", "exposure.npy")).astype(np.float32)
    out = _look(job, exposure, plate, mask, _look_opts(job))
    located = locate.Located.from_json(job["located"])
    score = fidelity.verify(piece, out, located)
    if score < FINAL_MIN_CORR:
        raise GateFailed(json.dumps({"final_verify": round(score, 4), "min": FINAL_MIN_CORR,
                                     "message": "the finished piece region does not match the original"}))
    png = media.save_image(job.file("out", f"{job['name']}-presentation.png"), out)
    jpg = media.save_image(job.file("out", f"{job['name']}-presentation.jpg"), out)
    manifest = {
        "piece": job["piece"], "prompt": job["prompt"], "scene": job["scene"], "energy": job["energy"],
        "aspect": job["aspect"], "resolution": job["resolution"], "size": [out.shape[1], out.shape[0]],
        "brief": job["brief"], "endpoints": job["endpoints"],
        "requests": {k: v.get("request_id") for k, v in job["fal"].items()},
        "gate": job["stages"]["still"], "final_verify": round(score, 4), "develop": job["develop"],
        "warnings": _warnings(job), "outputs": [str(png), str(jpg)],
    }
    job.file("out", "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    job.done("finish", final_verify=score)
    return {"presentation": str(png), "jpeg": str(jpg), "final_verify": round(score, 4),
            "warnings": _warnings(job)}


def status(job):
    return {"job": str(job.root), "stages": {s: job.is_done(s) for s in
            ["init", "analyze", "brief", "still", "video", "develop", "finish"]},
            "warnings": _warnings(job), "pending": {k: v["request_id"] for k, v in job["fal"].items()
                                                     if not v.get("done")}}
