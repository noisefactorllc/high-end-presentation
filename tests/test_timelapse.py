import cv2
import numpy as np
import pytest

from conftest import place, plain_scene, textured_piece, track_frame_liveness
from hep import color, fidelity, locate, media, timelapse


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


def _piece_scene():
    plate_bg = _scene_with_texture()
    piece = textured_piece(120, 160, seed=4)
    plate, H = place(piece, plate_bg, [[100, 60], [260, 60], [260, 180], [100, 180]])
    mask = fidelity.piece_mask(plate.shape, H, (160, 120))
    loc = locate.Located(True, H, locate.corners(160, 120), 0, 0, "")
    return piece, plate, mask, loc


def test_people_cross_the_piece_as_figures(tmp_path):
    piece, plate, mask, loc = _piece_scene()
    frames = []
    for i in range(40):
        f = plate.copy()
        if 0 < i < 39:
            x = 20 + i * 9  # a figure walking across, in front of the piece
            cv2.rectangle(f, (x, 40), (x + 30, 300), (0.05, 0.05, 0.05), -1)
        frames.append(f)
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    out, rep = timelapse.develop(plate, st, mask, "quiet")
    srgb = color.linear_to_srgb(out)
    inner = mask > 0.99
    # the figure shows over the piece, like everywhere else
    assert (plate[inner] - srgb[inner]).mean() > 0.02
    assert rep["piece_covered"] > 0
    # the uncovered parts of the piece are still the original
    assert fidelity.verify(piece, srgb, loc, exclude=rep["activity"].astype(np.float32)) > 0.9


def test_distortion_inside_the_frame_is_ignored(tmp_path):
    piece, plate, mask, loc = _piece_scene()
    frames = []
    for i in range(40):
        f = plate.copy()
        if 5 < i < 35:  # the video model "redraws" the artwork mid-clip
            f[70:170, 110:250] = np.roll(f[70:170, 110:250], 7 + i % 5, axis=1)
        frames.append(f)
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    out, _ = timelapse.develop(plate, st, mask, "bustling")
    srgb = color.linear_to_srgb(out)
    assert fidelity.verify(piece, srgb, loc) > 0.95


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


def test_density_is_solved_to_target():
    P = np.random.default_rng(0).beta(0.6, 8, (200, 200)).astype(np.float32)
    for target in (0.28, 0.52):
        k = timelapse.solve_density(P, target)
        act = P > 0.02
        assert abs((1 - (1 - P[act]) ** k).mean() - target) < 0.01


def test_single_frame_flash_fades_but_steady_light_stays(tmp_path):
    plate = _scene_with_texture()
    frames = []
    for i in range(30):
        f = plate.copy()
        cv2.circle(f, (60 + i * 20, 400), 4, (1, 1, 1), -1)   # fast: one frame per spot
        cv2.circle(f, (120 + i, 120), 4, (1, 1, 1), -1)       # slow: many frames per spot
        frames.append(f)
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    out, _ = timelapse.develop(plate, st, _full_mask(plate), "bustling")
    srgb = color.linear_to_srgb(out)
    fast = srgb[400, 60:360].mean() - plate[400, 60:360].mean()
    slow = srgb[120, 125:145].mean() - plate[120, 125:145].mean()
    assert slow > 2 * fast


def test_unmatched_reference_is_not_guessed():
    plate = _scene_with_texture()
    unrelated = np.full_like(plate, 0.4)
    with pytest.raises(ValueError, match="registration:initial"):
        timelapse.register(unrelated, plate)


def test_failed_jitter_stops_compositing(tmp_path, monkeypatch):
    plate = _scene_with_texture()
    monkeypatch.setattr(timelapse, "_jitter", lambda *args: (np.eye(3), False))
    with pytest.raises(ValueError, match="registration:frame:1"):
        timelapse.stack(_write(tmp_path / "v.mp4", [plate] * 4), plate)


def test_low_correlation_is_not_success(monkeypatch):
    def low_score(*args):
        return 0.2, np.eye(2, 3, dtype=np.float32)
    monkeypatch.setattr(cv2, "findTransformECC", low_score)
    g = np.random.default_rng(4).random((80, 80)).astype(np.float32)
    _, ok = timelapse._jitter(g, g, 1.0)
    assert not ok


@pytest.mark.parametrize("glazed", [False, True])
def test_crop_does_not_extend_moving_light_into_missing_border(tmp_path, glazed):
    plate = _scene_with_texture(480, 384)
    frames = []
    for i in range(24):
        f = plate[48:432, 40:344].copy()
        if 4 <= i < 20:
            cv2.rectangle(f, (0, 200), (12, 230), (1, 1, 1), -1)
        frames.append(f)
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    mask = np.ones(plate.shape[:2], np.float32) if glazed else _full_mask(plate)
    out, report = timelapse.develop(plate, st, mask, "bustling", glazed=glazed)
    diff = np.abs(out - color.srgb_to_linear(plate))
    assert diff[:, :30].max() < 1e-6
    assert diff[:35].max() < 1e-6
    assert diff[250:270, 44:50].mean() > 0.02  # real activity still survives
    assert 0.5 < report["registration_coverage"] < 0.8


def test_camera_rotation_is_measured_at_frame_corners(tmp_path):
    plate = _scene_with_texture()
    frames = []
    for angle in [0, 1.0, 2.0, 1.0, 0]:
        # Rotation about the origin has no translation component.
        M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        frames.append(cv2.warpAffine(plate, M, (384, 480), borderMode=cv2.BORDER_REFLECT))
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    assert st.moved
    assert st.max_shift > 0.02



def test_registration_keeps_background_fixed_while_person_and_camera_move(tmp_path):
    plate = _scene_with_texture()
    frames = []
    for i in range(24):
        f = plate.copy()
        if 3 <= i < 21:
            cv2.rectangle(f, (40 + i * 7, 200), (80 + i * 7, 360), (0.05, 0.05, 0.05), -1)
        W = cv2.getRotationMatrix2D((192, 240), np.sin(i) * 0.4, 1.0)
        W[:, 2] += [2 * np.sin(i), 1.5 * np.cos(i)]
        frames.append(cv2.warpAffine(f, W, (384, 480), borderMode=cv2.BORDER_REFLECT))
    st = timelapse.stack(_write(tmp_path / "v.mp4", frames), plate)
    out, _ = timelapse.develop(plate, st, _full_mask(plate), "calm")
    diff = np.abs(color.linear_to_srgb(out) - plate)
    assert np.percentile(diff[30:160, 30:-30], 99) < 0.03
    assert diff[210:350, 80:220].mean() > 0.02


def test_insufficient_shared_coverage_stops_compositing(tmp_path):
    plate = textured_piece(480, 384, seed=3)
    crop = plate[120:360, 96:288].copy()
    with pytest.raises(ValueError, match="registration:coverage"):
        timelapse.stack(_write(tmp_path / "v.mp4", [crop] * 4), plate)


def _moving_square_clip(path, n=42):
    """A textured piece with a white square crossing it, 4 px per frame."""
    base = textured_piece(160, 200, seed=9)
    frames = []
    for i in range(n):
        f = base.copy()
        f[60:80, 10 + 4 * i:30 + 4 * i] = 1.0
        frames.append(f)
    return _write(path, frames)


def _weight(out, frame, rows, cols):
    """Opacity of white over frame in out, in linear light."""
    o, b = color.srgb_to_linear(out[rows, cols]), color.srgb_to_linear(frame[rows, cols])
    return float(((o - b) / np.maximum(1 - b, 1e-3)).mean())


def test_piece_timelapse_layers_the_clip_behind_the_lead_frame(tmp_path):
    clip = _moving_square_clip(tmp_path / "p.mp4")
    out, used = timelapse.piece(clip, lead=21)
    assert used["lead"] == 21 and used["exposures"] == [3, 10, 17, 24, 31, 38]
    frames = [f for _, f in media.iter_frames(clip)]
    lin = [color.srgb_to_linear(f) for f in frames]
    share = timelapse.PIECE_LEAD
    want = share * lin[21] + (1 - share) * np.mean([lin[i] for i in used["exposures"]], 0)
    assert np.allclose(out, color.linear_to_srgb(want), atol=1e-4)
    # still parts keep the original detail
    assert fidelity.detail_correlation(out[100:], frames[0][100:]) > 0.98
    # the square shows where the lead frame holds it, and fainter along its path
    assert abs(_weight(out, frames[-1], slice(64, 76), slice(100, 106)) - share) < 0.05
    assert abs(_weight(out, frames[-1], slice(64, 76), slice(28, 36)) - (1 - share) / 6) < 0.03


def test_piece_timelapse_streams_frames(tmp_path, monkeypatch):
    clip = _moving_square_clip(tmp_path / "p.mp4")
    state = track_frame_liveness(monkeypatch, media)
    timelapse.piece(clip, lead=40)
    assert state["peak"] <= 4


def test_piece_timelapse_of_short_clips(tmp_path):
    out, used = timelapse.piece(_moving_square_clip(tmp_path / "three.mp4", n=3), lead=1)
    assert used["exposures"] == [0, 1, 2]
    one = _moving_square_clip(tmp_path / "one.mp4", n=1)
    out, used = timelapse.piece(one, lead=0)
    assert used["exposures"] == [0]
    assert np.abs(out - media.read_frame(one, 0)).max() < 1e-4


def test_piece_timelapse_needs_its_lead_frame(tmp_path):
    with pytest.raises(ValueError, match="frame 99"):
        timelapse.piece(_moving_square_clip(tmp_path / "p.mp4", n=5), lead=99)
