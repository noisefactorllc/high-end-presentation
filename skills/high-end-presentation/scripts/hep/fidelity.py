"""Separate scene lighting from piece content, gate on content drift, repair.

Lighting model: in linear light, reference ~= a * original + b per channel,
with a and b smooth fields fitted on a coarse grid. Whatever that model
cannot explain is content drift.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import color, locate, media

WORK_SIDE = 1536  # resolution for measuring (not for output)


@dataclass
class GateResult:
    passed: bool
    corr: float
    worst: float
    bands: list
    area: float
    located: locate.Located
    reason: str

    aspect_error: float = 0.0

    def to_json(self):
        return {"passed": self.passed, "corr": round(self.corr, 4), "worst_tile": round(self.worst, 4),
                "bands": [round(b, 4) for b in self.bands], "area": round(self.area, 4),
                "aspect_error": round(self.aspect_error, 4), "located": self.located.to_json(),
                "reason": self.reason}


def lighting_model(orig_lin, ref_lin, grid=12, eps=4e-4):
    """Smooth lighting fields (a, b) with ref ~= a * orig + b, per channel.

    Fit per tile on a coarse grid, reject outlier tiles with a median filter,
    then smooth and upsample. Scene light is smooth; anything sharper is the
    model changing content, and must not leak into the lighting."""
    h, w = orig_lin.shape[:2]
    gy = max(2, round(grid * h / max(h, w)))
    gx = max(2, round(grid * w / max(h, w)))
    A = np.ones((gy, gx, 3), np.float32)
    Bf = np.zeros((gy, gx, 3), np.float32)
    for i in range(gy):
        for j in range(gx):
            sl = (slice(i * h // gy, (i + 1) * h // gy), slice(j * w // gx, (j + 1) * w // gx))
            o = orig_lin[sl].reshape(-1, 3)
            r = ref_lin[sl].reshape(-1, 3)
            mo, mr = o.mean(0), r.mean(0)
            var = o.var(0)
            cov = ((o - mo) * (r - mr)).mean(0)
            a = cov / (var + eps)
            # flat tiles carry no slope information: fall back to a pure gain
            gain = (mr + 1e-4) / (mo + 1e-4)
            t = var / (var + eps)
            a = t * a + (1 - t) * gain
            a = np.clip(a, 0.15, 4.0)
            A[i, j] = a
            Bf[i, j] = np.clip(mr - a * mo, -0.05, 0.25)
    A = cv2.medianBlur(A, 3)
    Bf = cv2.medianBlur(Bf, 3)
    A = cv2.GaussianBlur(A, (0, 0), 0.8, borderType=cv2.BORDER_REPLICATE)
    Bf = cv2.GaussianBlur(Bf, (0, 0), 0.8, borderType=cv2.BORDER_REPLICATE)
    A = cv2.resize(A, (w, h), interpolation=cv2.INTER_CUBIC)
    Bf = cv2.resize(Bf, (w, h), interpolation=cv2.INTER_CUBIC)
    return A.astype(np.float32), Bf.astype(np.float32)


def _band(g, s1, s2):
    return cv2.GaussianBlur(g, (0, 0), s1) - cv2.GaussianBlur(g, (0, 0), s2)


def detail_bands(x, y, side=512):
    """Band-pass normalized correlation of luma at two scales."""
    h, w = x.shape[:2]
    s = side / max(h, w)
    size = (max(8, round(w * s)), max(8, round(h * s)))
    gx = color.luma(cv2.resize(x, size, interpolation=cv2.INTER_AREA))
    gy = color.luma(cv2.resize(y, size, interpolation=cv2.INTER_AREA))
    out = []
    for s1, s2 in ((0.8, 2.4), (2.4, 7.2)):
        bx, by = _band(gx, s1, s2), _band(gy, s1, s2)
        bx -= bx.mean()
        by -= by.mean()
        den = np.sqrt((bx * bx).sum() * (by * by).sum()) + 1e-12
        out.append(float((bx * by).sum() / den))
    return out


def detail_correlation(x, y):
    return float(np.mean(detail_bands(x, y)))


def tile_correlations(x, y, grid=4, side=512, min_energy=0.15):
    """Band-pass correlation per tile. Tiles with little detail in x (flat
    areas, where correlation means nothing) are skipped. Returns a list."""
    h, w = x.shape[:2]
    s = side / max(h, w)
    size = (max(8, round(w * s)), max(8, round(h * s)))
    gx = color.luma(cv2.resize(x, size, interpolation=cv2.INTER_AREA))
    gy = color.luma(cv2.resize(y, size, interpolation=cv2.INTER_AREA))
    bx = _band(gx, 0.8, 2.4) + _band(gx, 2.4, 7.2)
    by = _band(gy, 0.8, 2.4) + _band(gy, 2.4, 7.2)
    H, W = bx.shape
    tiles, energy = [], []
    for i in range(grid):
        for j in range(grid):
            sl = (slice(i * H // grid, (i + 1) * H // grid), slice(j * W // grid, (j + 1) * W // grid))
            tx, ty = bx[sl] - bx[sl].mean(), by[sl] - by[sl].mean()
            ex = float((tx * tx).sum())
            den = np.sqrt(ex * float((ty * ty).sum())) + 1e-12
            tiles.append(float((tx * ty).sum() / den))
            energy.append(ex)
    energy = np.array(energy)
    keep = energy >= min_energy * np.median(energy)
    return [t for t, k in zip(tiles, keep) if k]


def _work_piece(piece):
    h, w = piece.shape[:2]
    s = min(1.0, WORK_SIDE / max(h, w))
    return (media.resize(piece, round(w * s), round(h * s)) if s < 1 else piece), s


def measure(piece, scene, located):
    """Flatten the scene's copy of the piece at work resolution; return
    (work_piece, flat_ref, scale)."""
    wp, s = _work_piece(piece)
    Hs = locate.scaled_H(located.H, s)
    flat = locate.flatten(scene, Hs, (wp.shape[1], wp.shape[0]))
    return wp, flat, s


def aspect_error(piece, scene, located):
    """Relative error of the piece's apparent physical proportions."""
    est = locate.rectangle_aspect(located.quad, (scene.shape[1], scene.shape[0]))
    true = piece.shape[1] / piece.shape[0]
    return abs(est / true - 1.0)


def gate(piece, scene, *, min_corr=0.6, min_tile=0.35, min_area=0.02, max_aspect_error=0.04, located=None):
    located = located or locate.find_piece(piece, scene)
    area = 0.0
    if located.ok:
        area = locate.quad_area(located.quad) / float(scene.shape[0] * scene.shape[1])
    if not located.ok:
        return GateResult(False, 0.0, 0.0, [0.0, 0.0], area, located, f"not-found:{located.reason}")
    if area < min_area:
        return GateResult(False, 0.0, 0.0, [0.0, 0.0], area, located, "area")
    asp = aspect_error(piece, scene, located)
    if asp > max_aspect_error:
        return GateResult(False, 0.0, 0.0, [0.0, 0.0], area, located, "aspect", asp)
    wp, flat, _ = measure(piece, scene, located)
    a, b = lighting_model(color.srgb_to_linear(wp), color.srgb_to_linear(flat))
    relit = color.linear_to_srgb(a * color.srgb_to_linear(wp) + b)
    # Compare the scene's copy against the relit original: lighting explained,
    # only content differences remain.
    bands = detail_bands(relit, flat)
    corr = float(np.mean(bands))
    tiles = tile_correlations(relit, flat)
    worst = float(min(tiles)) if tiles else corr
    if corr < min_corr:
        return GateResult(False, corr, worst, bands, area, located, "detail-drift", asp)
    if worst < min_tile:
        return GateResult(False, corr, worst, bands, area, located, "local-drift", asp)
    return GateResult(True, corr, worst, bands, area, located, "", asp)


def piece_mask(shape, H, piece_size, feather=1.5, inset=1.0):
    """Soft mask of the piece region in scene pixels (1 inside)."""
    w, h = piece_size
    m = cv2.warpPerspective(np.ones((int(h), int(w)), np.float32), H, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
    if inset > 0:
        k = max(1, int(round(inset)))
        m = cv2.erode(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return np.clip(m, 0, 1)


def relight(piece, scene, located):
    """Return the original piece relit by the scene, at full piece resolution,
    in linear light, plus the (a, b) fields at full resolution."""
    wp, flat, _ = measure(piece, scene, located)
    a, b = lighting_model(color.srgb_to_linear(wp), color.srgb_to_linear(flat))
    h, w = piece.shape[:2]
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return a * color.srgb_to_linear(piece) + b, (a, b)


def warp_into(piece_lin, H, scene_shape):
    """Warp a (possibly much larger) piece image into scene pixels without aliasing."""
    h, w = piece_lin.shape[:2]
    footprint = np.sqrt(locate.quad_area(cv2.perspectiveTransform(locate.corners(w, h)[None], H)[0]) / (w * h))
    src, Hs = piece_lin, H
    if footprint < 0.75:
        s = min(1.0, footprint * 1.5)
        src = media.resize(piece_lin, max(2, round(w * s)), max(2, round(h * s)))
        Hs = locate.scaled_H(H, s)
    return cv2.warpPerspective(src, Hs, (scene_shape[1], scene_shape[0]), flags=cv2.INTER_LINEAR)


def repair(piece, scene, located):
    """Scene with the relit original pixels in the piece region.
    Returns (repaired_srgb, mask, relit_lin)."""
    relit_lin, _ = relight(piece, scene, located)
    warped = warp_into(relit_lin, located.H, scene.shape)
    mask = piece_mask(scene.shape, located.H, (piece.shape[1], piece.shape[0]))
    out_lin = color.srgb_to_linear(scene) * (1 - mask[..., None]) + warped * mask[..., None]
    return color.linear_to_srgb(out_lin), mask, relit_lin


def verify(piece, final, located):
    """Detail correlation between the original and the final image's piece region
    (lighting explained away first)."""
    wp, flat, _ = measure(piece, final, located)
    a, b = lighting_model(color.srgb_to_linear(wp), color.srgb_to_linear(flat))
    relit = color.linear_to_srgb(a * color.srgb_to_linear(wp) + b)
    return detail_correlation(relit, flat)
