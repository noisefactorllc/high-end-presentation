import cv2
import numpy as np

from conftest import place, textured_piece
from hep import color, fidelity


def _relit(img, gain, add):
    lin = color.srgb_to_linear(img)
    h, w = img.shape[:2]
    ramp = np.linspace(0.6, 1.2, w, dtype=np.float32)[None, :, None]
    return color.linear_to_srgb(lin * ramp * np.float32(gain) + np.float32(add))


def test_identical_passes(piece, scene, quad):
    composed, _ = place(piece, scene, quad)
    g = fidelity.gate(piece, composed)
    assert g.passed, g.reason
    assert g.corr > 0.85


def test_relit_passes(piece, scene, quad):
    composed, _ = place(_relit(piece, [1.0, 0.85, 0.6], [0.01, 0.005, 0.0]), scene, quad)
    g = fidelity.gate(piece, composed)
    assert g.passed, g.reason
    assert g.corr > 0.8


def test_redrawn_texture_passes_but_new_composition_fails(piece, scene, quad):
    other = textured_piece(seed=9)
    # fine texture redrawn everywhere: the gate accepts it (the repair restores the pixels)
    retextured = cv2.GaussianBlur(piece, (0, 0), 3) + (other - cv2.GaussianBlur(other, (0, 0), 3))
    composed, _ = place(np.clip(retextured, 0, 1), scene, quad)
    assert fidelity.gate(piece, composed).passed
    # the right third replaced by different content: the gate rejects it
    mixed = piece.copy()
    x0 = piece.shape[1] * 2 // 3
    noise = np.random.default_rng(3).random(piece.shape).astype(np.float32)
    unrelated = np.clip(0.5 + 3.0 * (cv2.GaussianBlur(noise, (0, 0), 8) - 0.5), 0, 1)
    mixed[:, x0:] = unrelated[:, x0:]
    composed, _ = place(mixed, scene, quad)
    g = fidelity.gate(piece, composed)
    assert g.located.ok
    assert not g.passed and g.reason in ("composition", "local-composition")


def test_wrong_colours_fail(piece, scene, quad):
    warm = np.clip(piece * np.float32([1.0, 0.55, 0.25]), 0, 1)  # an orange-dominant piece
    cool = warm[..., ::-1].copy()  # same structure, blue-dominant
    composed, _ = place(cool, scene, quad)
    g = fidelity.gate(warm, composed)
    assert not g.passed and g.reason == "color"


def test_small_piece_fails_on_area(piece, scene):
    q = [[300, 300], [345, 300], [345, 334], [300, 334]]
    composed, _ = place(piece, scene, q)
    g = fidelity.gate(piece, composed, min_area=0.02)
    assert not g.passed
    assert g.reason == "area" or g.reason.startswith("not-found")


def test_repair_restores_original_detail(piece, scene, quad):
    soft = cv2.GaussianBlur(_relit(piece, [1.0, 0.9, 0.7], 0.0), (0, 0), 2.0)
    composed, _ = place(soft, scene, quad)
    g = fidelity.gate(piece, composed, min_corr=0.0)
    repaired, mask, _ = fidelity.repair(piece, composed, g.located)
    assert fidelity.verify(piece, repaired, g.located) > 0.95
    assert fidelity.verify(piece, composed, g.located) < fidelity.verify(piece, repaired, g.located)
    # outside the piece, the scene is untouched
    outside = mask < 1e-4
    assert np.abs(repaired[outside] - composed[outside]).max() < 2e-3
    # the repaired piece keeps the scene's warm light
    inside = mask > 0.99
    r, b = repaired[inside][:, 0].mean(), repaired[inside][:, 2].mean()
    assert r > b


def test_squeezed_piece_fails_on_aspect(piece, scene):
    h, w = piece.shape[:2]
    squeezed = cv2.resize(piece, (int(w * 0.9), h))
    # placed frontally with the wrong proportions
    q = [[150, 250], [150 + int(w * 0.9), 250], [150 + int(w * 0.9), 250 + h], [150, 250 + h]]
    composed, _ = place(squeezed, scene, q)
    g = fidelity.gate(piece, composed)
    assert g.located.ok
    assert not g.passed and g.reason == "aspect"


def test_glazed_repair_keeps_glass_reflection(piece, scene, quad):
    composed, H = place(piece, scene, quad)
    # a soft streetlight reflection on the glass over the piece
    h, w = composed.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c = np.mean(quad, 0)
    glow = 0.35 * np.exp(-(((xx - c[0]) / 40) ** 2 + ((yy - c[1]) / 30) ** 2))
    lit = color.linear_to_srgb(color.srgb_to_linear(composed) + glow[..., None])
    g = fidelity.gate(piece, lit)
    plain, _, _ = fidelity.repair(piece, lit, g.located, glazed=False)
    glazed, _, _ = fidelity.repair(piece, lit, g.located, glazed=True)
    cy, cx = int(c[1]), int(c[0])
    spot = (slice(cy - 10, cy + 10), slice(cx - 10, cx + 10))
    assert glazed[spot].mean() > plain[spot].mean() + 0.03
    # the reflection is soft light: the piece's detail is still the original
    assert fidelity.verify(piece, glazed, g.located) > 0.85
