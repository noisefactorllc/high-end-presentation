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
# The gate compares composition and colour: image models redraw fine texture,
# and the repair restores the original pixels anyway. The final check compares
# fine detail (these band-pass scales, Gaussian sigmas at 512 px), because by
# then every pixel of the piece must be the artist's.
FINE_BANDS = ((0.8, 2.4), (2.4, 7.2))
COMP_SIDE = 384
COMP_SIGMAS = (6.0, 16.0)  # blur scales for composition, at COMP_SIDE


@dataclass
class GateResult:
    passed: bool
    corr: float
    worst: float
    color_shift: float
    area: float
    located: locate.Located
    reason: str
    aspect_error: float = 0.0

    def to_json(self):
        return {"passed": self.passed, "corr": round(self.corr, 4), "worst_tile": round(self.worst, 4),
                "color_shift": round(self.color_shift, 2), "area": round(self.area, 4),
                "aspect_error": round(self.aspect_error, 4), "located": self.located.to_json(),
                "reason": self.reason}


def lighting_model(orig_lin, ref_lin, grid=12, eps=4e-4):
    """Smooth lighting fields (a, b) with ref ~= a * orig + b, per channel.

    Per tile, match the mean and spread of the blurred images (moment matching,
    not regression): it does not depend on the two images lining up exactly,
    so a model that shifted or zoomed the piece cannot read as a loss of
    contrast. Outlier tiles are rejected with a median filter; the fields are
    smoothed and upsampled. Scene light is smooth; sharper differences are the
    model changing content and must not leak into the lighting."""
    h, w = orig_lin.shape[:2]
    sig = max(1.0, 0.006 * max(h, w))
    orig_lin = cv2.GaussianBlur(orig_lin, (0, 0), sig)
    ref_lin = cv2.GaussianBlur(ref_lin, (0, 0), sig)
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
            so, sr = o.std(0), r.std(0)
            a = np.sqrt((sr * sr + eps) / (so * so + eps))
            # flat tiles carry no contrast information: fall back to a pure gain
            gain = (mr + 1e-4) / (mo + 1e-4)
            t = so * so / (so * so + eps)
            a = np.clip(t * a + (1 - t) * gain, 0.15, 4.0)
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


def detail_bands(x, y, side=512, bands=FINE_BANDS):
    """Band-pass normalized correlation of luma at two scales."""
    h, w = x.shape[:2]
    s = side / max(h, w)
    size = (max(8, round(w * s)), max(8, round(h * s)))
    gx = color.luma(cv2.resize(x, size, interpolation=cv2.INTER_AREA))
    gy = color.luma(cv2.resize(y, size, interpolation=cv2.INTER_AREA))
    out = []
    for s1, s2 in bands:
        bx, by = _band(gx, s1, s2), _band(gy, s1, s2)
        bx -= bx.mean()
        by -= by.mean()
        den = np.sqrt((bx * bx).sum() * (by * by).sum()) + 1e-12
        out.append(float((bx * by).sum() / den))
    return out


def detail_correlation(x, y, bands=FINE_BANDS):
    return float(np.mean(detail_bands(x, y, bands=bands)))


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


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def composition(piece, scene, H, side=COMP_SIDE, grid=3):
    """How well the scene under H shows this piece, ignoring texture and light:
    (correlation of blurred luma, worst tile correlation, colour distance)."""
    h, w = piece.shape[:2]
    s = side / max(h, w)
    small = media.resize(piece, max(8, round(w * s)), max(8, round(h * s)))
    flat = locate.flatten(scene, locate.scaled_H(H, s), (small.shape[1], small.shape[0]))
    lp, lf = color.luma(small), color.luma(flat)
    corr = float(np.mean([_ncc(cv2.GaussianBlur(lp, (0, 0), g), cv2.GaussianBlur(lf, (0, 0), g))
                          for g in COMP_SIGMAS]))
    bp, bf = cv2.GaussianBlur(lp, (0, 0), COMP_SIGMAS[0]), cv2.GaussianBlur(lf, (0, 0), COMP_SIGMAS[0])
    hh, ww = bp.shape
    tiles = []
    for i in range(grid):
        for j in range(grid):
            sl = (slice(i * hh // grid, (i + 1) * hh // grid), slice(j * ww // grid, (j + 1) * ww // grid))
            if bp[sl].std() > 0.01:  # flat tiles say nothing about composition
                tiles.append(_ncc(bp[sl], bf[sl]))
    worst = min(tiles) if tiles else corr
    ab_p = color.rgb_to_lab(small)[..., 1:].reshape(-1, 2).mean(0)
    ab_f = color.rgb_to_lab(flat)[..., 1:].reshape(-1, 2).mean(0)
    return corr, float(worst), float(np.linalg.norm(ab_p - ab_f))


def _score(piece, scene, H):
    corr, _, dab = composition(piece, scene, H)
    return corr - max(0.0, (dab - 20.0) / 40.0)


def locate_piece(piece, scene, min_inliers=12, top=3):
    """Find the piece: feature matches first, then rectangles in the scene
    (screens, frames) as candidates; refine each by ECC and keep the placement
    whose detail matches best."""
    h, w = piece.shape[:2]
    cands = []
    sift = locate.find_piece(piece, scene, min_inliers=min_inliers)
    if sift.ok:
        cands.append(("sift", sift.H, sift.inliers, sift.matches))
    quads = locate.candidate_quads(scene)
    ranked = []
    for q in quads:
        H = cv2.getPerspectiveTransform(locate.corners(w, h).astype(np.float32), np.float32(q)).astype(np.float64)
        ranked.append((_score(piece, scene, H), H))
    ranked.sort(key=lambda t: -t[0])
    cands += [("quad", H, 0, 0) for _, H in ranked[:top]]
    best = None
    for how, H, inl, mat in cands:
        Hr, ok = locate.refine(piece, scene, H)
        for Hc in ((Hr, H) if ok else (H,)):
            quad = cv2.perspectiveTransform(locate.corners(w, h)[None], Hc)[0]
            if not locate.is_convex(quad):
                continue
            sc = _score(piece, scene, Hc)
            if best is None or sc > best[0]:
                best = (sc, locate.Located(True, Hc, quad, inl, mat, how))
    if best is None:
        return locate.Located(False, reason=sift.reason or "no-candidates")
    return best[1]


def aspect_error(piece, scene, located):
    """Relative error of the piece's apparent physical proportions."""
    est = locate.rectangle_aspect(located.quad, (scene.shape[1], scene.shape[0]))
    true = piece.shape[1] / piece.shape[0]
    return abs(est / true - 1.0)


def gate(piece, scene, *, min_corr=0.5, min_tile=0.15, max_color_shift=35.0, min_area=0.02,
         max_aspect_error=0.04, located=None):
    """Is this the piece, whole, in the right proportions? Checks placement,
    composition, and colour. Fine texture is not checked: the repair restores
    the original pixels, and the final check verifies them."""
    located = located or locate_piece(piece, scene)
    area = 0.0
    if located.ok:
        area = locate.quad_area(located.quad) / float(scene.shape[0] * scene.shape[1])
    if not located.ok:
        return GateResult(False, 0.0, 0.0, 0.0, area, located, f"not-found:{located.reason}")
    if area < min_area:
        return GateResult(False, 0.0, 0.0, 0.0, area, located, "area")
    asp = aspect_error(piece, scene, located)
    corr, worst, dab = composition(piece, scene, located.H)
    for failed, reason in ((asp > max_aspect_error, "aspect"), (dab > max_color_shift, "color"),
                           (corr < min_corr, "composition"), (worst < min_tile, "local-composition")):
        if failed:
            return GateResult(False, corr, worst, dab, area, located, reason, asp)
    return GateResult(True, corr, worst, dab, area, located, "", asp)


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


REFLECTION_SIGMA = 0.08   # of the piece's long side: glass reflections are soft
REFLECTION_STRENGTH = 0.6


def reflection(piece, scene, located):
    """Soft light the scene adds over the piece (glass reflections), at full
    piece resolution, linear. Only the broad, positive part of what the lighting
    model leaves unexplained: a heavy blur keeps the model's own version of the
    piece (redrawn or shifted content) from passing through."""
    wp, flat, _ = measure(piece, scene, located)
    orig_lin, ref_lin = color.srgb_to_linear(wp), color.srgb_to_linear(flat)
    a, b = lighting_model(orig_lin, ref_lin)
    R = cv2.GaussianBlur(ref_lin - (a * orig_lin + b), (0, 0), REFLECTION_SIGMA * max(wp.shape[:2]))
    R = np.maximum(R, 0) * REFLECTION_STRENGTH
    h, w = piece.shape[:2]
    return cv2.resize(R, (w, h), interpolation=cv2.INTER_LINEAR)


def repair(piece, scene, located, glazed=False):
    """Scene with the relit original pixels in the piece region. For a glazed
    display, the scene's glass reflections stay over the piece.
    Returns (repaired_srgb, mask, relit_lin)."""
    relit_lin, _ = relight(piece, scene, located)
    if glazed:
        relit_lin = relit_lin + reflection(piece, scene, located)
    warped = warp_into(relit_lin, located.H, scene.shape)
    mask = piece_mask(scene.shape, located.H, (piece.shape[1], piece.shape[0]))
    out_lin = color.srgb_to_linear(scene) * (1 - mask[..., None]) + warped * mask[..., None]
    return color.linear_to_srgb(out_lin), mask, relit_lin


def verify(piece, final, located, exclude=None):
    """Detail correlation between the original and the final image's piece
    region, with lighting explained away first. `exclude` (scene-sized, 0-1)
    marks pixels where people or reflected light legitimately cover the piece;
    those pixels are left out of the comparison."""
    wp, flat, s = measure(piece, final, located)
    a, b = lighting_model(color.srgb_to_linear(wp), color.srgb_to_linear(flat))
    relit = color.linear_to_srgb(a * color.srgb_to_linear(wp) + b)
    if exclude is not None:
        Hs = locate.scaled_H(located.H, s)
        m = locate.flatten(np.asarray(exclude, np.float32), Hs, (wp.shape[1], wp.shape[0]))
        m = np.clip(cv2.GaussianBlur(m, (0, 0), 2.0) * 2.0, 0, 1)[..., None]
        flat = flat * (1 - m) + relit * m
    return detail_correlation(relit, flat)
