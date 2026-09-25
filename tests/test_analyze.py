import cv2
import numpy as np

from hep import analyze


def test_palette_finds_two_colors_by_area():
    img = np.zeros((100, 100, 3), np.float32)
    img[:, :70] = [0.9, 0.1, 0.1]
    img[:, 70:] = [0.1, 0.2, 0.9]
    pal = analyze.palette(img, k=4)
    assert len(pal) == 2
    assert pal[0]["hex"].startswith("#e") and abs(pal[0]["weight"] - 0.7) < 0.02
    assert abs(pal[1]["weight"] - 0.3) < 0.02


def test_warmth_orders_warm_over_cool():
    warm = np.full((20, 20, 3), [0.9, 0.55, 0.2], np.float32)
    cool = np.full((20, 20, 3), [0.2, 0.5, 0.9], np.float32)
    assert analyze.tone_stats(warm)["warmth"] > analyze.tone_stats(cool)["warmth"]


def test_representative_frame_skips_black_and_blur(tmp_path, piece):
    p = tmp_path / "v.mp4"
    h, w = piece.shape[:2]
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    sharp = (piece[..., ::-1] * 255).astype(np.uint8)
    for i in range(48):
        if i < 16:
            f = np.zeros_like(sharp)
        elif i < 32:
            f = cv2.GaussianBlur(sharp, (0, 0), 8)
        else:
            f = sharp
        vw.write(f)
    vw.release()
    index, frame = analyze.representative_frame(p, samples=24)
    assert index >= 32


def test_analyze_piece_still(tmp_path, piece):
    from hep import media
    p = media.save_image(tmp_path / "piece.png", piece)
    result, still = analyze.analyze_piece(p)
    assert result["source"]["kind"] == "image"
    assert still.shape == piece.shape
    assert len(result["palette"]) == 6
    assert set(result["tone"]) == {"key", "contrast", "warmth", "saturation"}
