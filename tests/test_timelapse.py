import cv2
import numpy as np
import pytest

from conftest import place, plain_scene, textured_piece
from hep import color, fidelity, locate, timelapse


def _scene_with_texture(h=480, w=384):
    # textured backdrop so frame registration has features
    s = plain_scene(h, w)
    tex = textured_piece(h, w, seed=3)
    return np.clip(0.6 * s + 0.4 * tex, 0, 1).astype(np.float32)


def _write(path, frames, fps=24):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write((np.clip(f, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))
    vw.release()
    return path


def _full_mask(plate):
    return np.zeros(plate.shape[:2], np.float32)


def test_static_video_gives_no_delta(tmp_path):
    plate = _scene_with_texture()
    p = _write(tmp_path / "v.mp4", [plate] * 24)
    st = timelapse.stack(p, plate)
    out, rep = timelapse.develop(plate, st, _full_mask(plate), "calm")
    diff = np.abs(color.linear_to_srgb(out) - plate)
    assert np.percentile(diff, 99) < 0.03
    assert any("no-motion" in n for n in rep["notes"])


def test_moving_light_leaves_trail(tmp_path):
    plate = _scene_with_texture()
    frames = []
    for i in range(48):
        f = plate.copy()
        cv2.circle(f, (40 + i * 6, 380), 5, (1, 1, 1), -1)
        frames.append(f)
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    out, _ = timelapse.develop(plate, st, _full_mask(plate), "lively")
    srgb = color.linear_to_srgb(out)
    on_path = srgb[380, 60:300].mean()
    off_path = srgb[300, 60:300].mean()
    assert on_path - plate[380, 60:300].mean() > 0.08
    assert abs(off_path - plate[300, 60:300].mean()) < 0.02


def test_stationary_viewer_more_solid_than_passerby(tmp_path):
    plate = _scene_with_texture()
    frames = []
    for i in range(50):
        f = plate.copy()
        if 5 <= i < 35:  # the viewer stands for 60% of the clip
            cv2.rectangle(f, (60, 200), (100, 330), (0.1, 0.1, 0.12), -1)
        if 20 <= i < 25:  # a passer-by stands in one place for 10%
            cv2.rectangle(f, (260, 200), (300, 330), (0.1, 0.1, 0.12), -1)
        frames.append(f)
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    out, _ = timelapse.develop(plate, st, _full_mask(plate), "calm")
    srgb = color.linear_to_srgb(out)
    dark_viewer = plate[210:320, 65:95].mean() - srgb[210:320, 65:95].mean()
    dark_passer = plate[210:320, 265:295].mean() - srgb[210:320, 265:295].mean()
    assert dark_viewer > 2 * dark_passer > 0


def test_quiet_keeps_piece_detail(tmp_path):
    plate_bg = _scene_with_texture()
    piece = textured_piece(120, 160, seed=4)
    plate, H = place(piece, plate_bg, [[100, 60], [260, 60], [260, 180], [100, 180]])
    mask = fidelity.piece_mask(plate.shape, H, (160, 120))
    frames = []
    for i in range(40):
        f = plate.copy()
        x = 20 + i * 9  # a figure walking across, in front of the piece
        cv2.rectangle(f, (x, 40), (x + 30, 200), (0.05, 0.05, 0.05), -1)
        frames.append(f)
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    out, rep = timelapse.develop(plate, st, mask, "quiet")
    srgb = color.linear_to_srgb(out)
    inner = mask > 0.99
    # detail inside the piece is the plate's (lighting only), not the walker's
    loc = locate.Located(True, H, locate.corners(160, 120), 0, 0, "")
    assert fidelity.verify(piece, srgb, loc) > 0.9
    assert rep["occlusion_raw"] > 0 and any("occlusion-removed" in n for n in rep["notes"])
    # but bustling lets a faint ghost through
    out_b, _ = timelapse.develop(plate, st, mask, "bustling")
    assert np.abs(color.linear_to_srgb(out_b)[inner] - srgb[inner]).mean() > 0.002


def test_registration_handles_scaled_cropped_video(tmp_path):
    plate = _scene_with_texture(600, 480)
    # the "video" is a 0.6x downscale of a centered crop of the plate
    crop = plate[30:570, 24:456]
    small = cv2.resize(crop, (259, 324), interpolation=cv2.INTER_AREA)
    frames = []
    for i in range(30):
        f = small.copy()
        cv2.circle(f, (20 + i * 7, 250), 4, (1, 1, 1), -1)
        frames.append(f)
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    assert "registration:sift" in st.notes
    # the background lines up with the plate on the canvas
    canvas_plate = cv2.resize(plate, (st.B.shape[1], st.B.shape[0]), interpolation=cv2.INTER_AREA)
    b = color.linear_to_srgb(st.B)
    ys, xs = slice(int(0.2 * b.shape[0]), int(0.8 * b.shape[0])), slice(int(0.2 * b.shape[1]), int(0.8 * b.shape[1]))
    assert np.abs(b[ys, xs] - canvas_plate[ys, xs]).mean() < 0.03


def test_jitter_is_corrected_and_measured(tmp_path):
    plate = _scene_with_texture()
    frames = []
    for i in range(20):
        M = np.float32([[1, 0, 2.0 * np.sin(i)], [0, 1, 1.5 * np.cos(i)]])
        frames.append(cv2.warpAffine(plate, M, (plate.shape[1], plate.shape[0]), borderMode=cv2.BORDER_REPLICATE))
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    assert not st.moved
    b = color.linear_to_srgb(st.B)
    canvas_plate = cv2.resize(plate, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(b - canvas_plate)[20:-20, 20:-20].mean() < 0.025
