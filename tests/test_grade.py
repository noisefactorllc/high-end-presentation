import numpy as np

from conftest import textured_piece
from hep import analyze, color, grade


def test_tone_map_identity_in_range_and_compresses_highlights():
    x = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32) * 0.9
    assert np.allclose(grade.tone_map(color.srgb_to_linear(x)), x, atol=2e-3)
    hot = color.srgb_to_linear(x).copy()
    hot[:4] *= 6.0
    out = grade.tone_map(hot)
    assert out.max() <= 1.0 and out[:4].mean() > out[8:].mean()


def test_tone_map_monotonic():
    ramp = np.linspace(0, 4, 256, dtype=np.float32)[None, :, None].repeat(3, -1).repeat(4, 0)
    out = grade.tone_map(ramp)[0, :, 0]
    assert np.all(np.diff(out) >= -1e-6)


def test_piece_changes_less_than_surround():
    img = textured_piece(200, 200, seed=6)
    pal = analyze.palette(np.full((10, 10, 3), [0.8, 0.3, 0.2], np.float32), k=1) + \
        analyze.palette(np.full((10, 10, 3), [0.1, 0.2, 0.5], np.float32), k=1)
    mask = np.zeros((200, 200), np.float32)
    mask[50:150, 50:150] = 1
    lin = color.srgb_to_linear(img)
    out = grade.apply(lin, pal, mask, tone=0.4, grain_amount=0.0)
    d = np.abs(out - img).mean(-1)
    assert d[mask > 0.5].mean() < 0.75 * d[mask < 0.5].mean()


def test_split_tone_without_chroma_is_noop():
    img = np.full((8, 8, 3), 0.4, np.float32)
    gray_pal = [{"lab": [50, 0, 0], "weight": 1.0}]
    assert np.allclose(grade.split_tone(img, gray_pal, 0.5), img, atol=2e-3)
