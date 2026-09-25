import cv2
import numpy as np

from conftest import textured_piece
from hep import lens


def _setup():
    img = textured_piece(240, 320, seed=2)
    depth = np.tile(np.linspace(0, 1, 320, dtype=np.float32), (240, 1))  # far left, near right
    mask = np.zeros((240, 320), np.float32)
    mask[80:160, 140:200] = 1.0  # piece in the middle
    return img, depth, mask


def test_piece_region_is_exact_and_far_region_blurs():
    img, depth, mask = _setup()
    out, coc = lens.depth_of_field(img, depth, mask, strength=1.0)
    assert np.array_equal(out[mask > 0.999], img[mask > 0.999])
    far = (slice(20, 220), slice(0, 60))
    hf = lambda a: cv2.Laplacian(a.mean(-1), cv2.CV_32F).var()
    assert hf(out[far]) < 0.5 * hf(img[far])
    assert coc[:, :40].mean() > coc[:, 150:190].mean()


def test_zero_strength_is_identity():
    img, depth, mask = _setup()
    out, _ = lens.depth_of_field(img, depth, mask, strength=0.0)
    assert np.allclose(out, img)


def test_vignette_darkens_corners_not_center():
    img = np.full((100, 120, 3), 0.5, np.float32)
    v = lens.vignette(img, 0.3)
    assert v[0, 0, 0] < 0.45 and abs(v[50, 60, 0] - 0.5) < 1e-3
