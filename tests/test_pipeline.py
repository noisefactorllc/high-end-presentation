"""Full stage chain with fal replaced by a local fake."""
import base64
import json

import cv2
import numpy as np
import pytest

import hep as _pkg  # noqa: F401  (conftest puts scripts/ on the path)
from conftest import place, plain_scene, textured_piece
from hep import media, stages
from hep.job import Job, JobError
import importlib.util
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "skills/high-end-presentation/scripts/hep.py"


def _png_uri(img):
    ok, buf = cv2.imencode(".png", (np.clip(img, 0, 1)[..., ::-1] * 255).astype(np.uint8))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class FakeClient:
    def __init__(self, tmp, scene_img):
        self.tmp, self.scene_img, self.calls = tmp, scene_img, []

    def run(self, endpoint, args, state, slot, timeout, save=lambda: None):
        self.calls.append(endpoint)
        state[slot] = {"endpoint": endpoint, "request_id": f"fake-{slot}", "done": True}
        save()
        if "edit" in endpoint:
            return {"images": [{"url": _png_uri(self.scene_img)}]}
        if "depth" in endpoint:
            h, w = self.scene_img.shape[:2]
            d = np.tile(np.linspace(0.2, 0.9, h, dtype=np.float32)[:, None], (1, w))
            return {"image": {"url": _png_uri(np.repeat(d[..., None], 3, -1))}}
        # video: the scene with a walker passing and a light moving
        p = self.tmp / "fake.mp4"
        h, w = self.scene_img.shape[:2]
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
        for i in range(36):
            f = self.scene_img.copy()
            if 0 < i < 35:
                x = 10 + i * 6
                cv2.rectangle(f, (x, h - 260), (x + 28, h - 60), (0.08, 0.08, 0.1), -1)
                cv2.circle(f, (w - 30 - i * 5, h - 30), 5, (1, 1, 0.9), -1)
            vw.write((f[..., ::-1] * 255 + 0.5).astype(np.uint8))
        vw.release()
        return {"video": {"url": "data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode()}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    piece = textured_piece(300, 400, seed=7)
    bg = np.clip(0.6 * plain_scene(640, 512) + 0.4 * textured_piece(640, 512, seed=3), 0, 1).astype(np.float32)
    scene, _ = place(cv2.GaussianBlur(piece, (0, 0), 0.8), bg, [[96, 120], [416, 130], [410, 370], [100, 362]])
    fake = FakeClient(tmp_path, scene)
    monkeypatch.setattr(stages, "_client", lambda: fake)
    pp = media.save_image(tmp_path / "art.png", piece)
    return tmp_path, pp, fake


def _brief(tmp):
    b = tmp / "brief.json"
    b.write_text(json.dumps({"themes": ["flow"], "mood": "calm", "scene": "A quiet gallery wall.",
                             "motion": "", "focus": 1.0}))
    return b


def test_full_chain(setup):
    tmp, pp, fake = setup
    root = tmp / "job"
    stages.init(root, pp, "gallery", energy="calm")
    job = Job(root)
    stages.analyze(job)
    stages.brief(job, json.loads(_brief(tmp).read_text()))
    s = stages.still(job)
    assert s["passed"]
    assert Path(stages.draft(job)["draft"]).exists()
    stages.video(job)
    rep = stages.develop(job)["report"]
    assert rep["frames"] == 36
    out = stages.finish(job)
    assert out["final_verify"] >= stages.FINAL_MIN_CORR
    manifest = json.loads((root / "out/manifest.json").read_text())
    assert manifest["requests"]["video"] == "fake-video"
    img = media.load_image(out["jpeg"])
    assert img.shape[:2] == (640, 512)
    # the depth model was called once and cached for finish
    assert fake.calls.count("fal-ai/image-preprocessors/depth-anything/v2") == 1


def test_stage_order_enforced(setup):
    tmp, pp, _ = setup
    root = tmp / "job"
    stages.init(root, pp, "x")
    with pytest.raises(JobError, match="analyze"):
        stages.still(Job(root))


def test_brief_requires_fields(setup):
    tmp, pp, _ = setup
    stages.init(tmp / "job", pp, "x")
    job = Job(tmp / "job")
    stages.analyze(job)
    with pytest.raises(JobError, match="missing"):
        stages.brief(job, {"themes": []})


def test_gate_failure_exit_code(setup, monkeypatch, capsys):
    tmp, pp, fake = setup
    fake.scene_img = plain_scene(640, 512)  # the model "lost" the piece
    spec = importlib.util.spec_from_file_location("runner", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    root = str(tmp / "job")
    assert runner.main(["init", "--job", root, "--piece", str(pp)]) == 0
    assert runner.main(["analyze", "--job", root]) == 0
    assert runner.main(["brief", "--job", root, "--brief-file", str(_brief(tmp))]) == 0
    capsys.readouterr()
    assert runner.main(["still", "--job", root, "--attempts", "2"]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "fidelity-gate" and len(out["detail"]["attempts"]) == 2


def test_bad_inputs_rejected(tmp_path):
    with pytest.raises(JobError, match="energy"):
        stages.init(tmp_path / "j", __file__, "x", energy="frantic")
    with pytest.raises(JobError, match="not found"):
        stages.init(tmp_path / "j", tmp_path / "nope.png", "x")


def test_moving_piece_with_echo(setup):
    tmp, _, fake = setup
    base = textured_piece(300, 400, seed=7)
    clip = tmp / "art.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 24, (400, 300))
    for i in range(48):
        f = np.roll(base, i * 2, axis=1) if i >= 24 else base  # holds still, then drifts
        vw.write((f[..., ::-1] * 255 + 0.5).astype(np.uint8))
    vw.release()
    root = tmp / "vjob"
    stages.init(root, clip, "gallery", echo=True)
    job = Job(root)
    a = stages.analyze(job)["analysis"]
    assert a["source"]["kind"] == "video"
    assert (root / "piece-echo.png").exists() and (root / "piece-preview.jpg").exists()
    stages.brief(job, json.loads(_brief(tmp).read_text()))
    assert stages.still(job)["passed"]
    stages.video(job)
    stages.develop(job)
    assert stages.finish(job)["final_verify"] >= stages.FINAL_MIN_CORR


def test_explicit_frame_is_used(setup):
    tmp, _, _ = setup
    clip = tmp / "c.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    for i in range(10):
        vw.write(np.full((48, 64, 3), i * 25, np.uint8))
    vw.release()
    stages.init(tmp / "fj", clip, "x", frame=7)
    job = Job(tmp / "fj")
    stages.analyze(job)
    assert job["analysis"]["source"]["frame_index"] == 7
    assert abs(media.load_image(tmp / "fj/piece.png").mean() - 175 / 255) < 0.02
