import cv2
import numpy as np
from PIL import Image

from hep import color, media


def test_srgb_linear_roundtrip():
    a = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.allclose(color.linear_to_srgb(color.srgb_to_linear(a)), a, atol=1e-5)


def test_lab_roundtrip():
    rgb = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    assert np.allclose(color.lab_to_rgb(color.rgb_to_lab(rgb)), rgb, atol=2e-3)


def test_rgba_composites_over_black(tmp_path):
    a = np.zeros((4, 4, 4), np.uint8)
    a[..., 0] = 255
    a[..., 3] = 128
    p = tmp_path / "a.png"
    Image.fromarray(a, "RGBA").save(p)
    img = media.load_image(p)
    assert img.shape == (4, 4, 3)
    assert abs(img[0, 0, 0] - 128 / 255) < 1e-3 and img[0, 0, 1] == 0


def test_16bit_png_roundtrip(tmp_path):
    rgb = np.random.default_rng(1).random((8, 8, 3)).astype(np.float32)
    p = media.save_image(tmp_path / "x.png", rgb)
    back = media.load_image_any_depth(p)
    assert back.shape == (8, 8, 3)
    assert np.allclose(back, rgb, atol=1 / 60000)


def test_grayscale_and_palette_load(tmp_path):
    Image.fromarray(np.full((5, 6), 200, np.uint8), "L").save(tmp_path / "g.png")
    Image.fromarray(np.full((5, 6), 200, np.uint8), "L").convert("P").save(tmp_path / "p.png")
    for name in ("g.png", "p.png"):
        assert media.load_image(tmp_path / name).shape == (5, 6, 3)


def test_video_roundtrip(tmp_path):
    p = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    for i in range(10):
        f = np.full((48, 64, 3), i * 20, np.uint8)
        vw.write(f)
    vw.release()
    info = media.video_info(p)
    assert (info["width"], info["height"]) == (64, 48)
    frames = list(media.iter_frames(p))
    assert len(frames) == 10
    assert frames[5][1].shape == (48, 64, 3)
    assert media.is_video(p) and not media.is_video(tmp_path / "x.png")
