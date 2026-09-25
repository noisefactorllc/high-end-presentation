"""Depth of field and optical vignette, in linear light."""
import cv2
import numpy as np

LAYERS = 8


def _disc(r):
    if r < 0.5:
        return None
    k = int(np.ceil(r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1].astype(np.float32)
    d = np.clip(r + 0.5 - np.sqrt(xx * xx + yy * yy), 0, 1)  # anti-aliased disc
    return d / d.sum()


def focus_depth(depth, piece_mask):
    sel = piece_mask > 0.5
    return float(np.median(depth[sel])) if sel.any() else float(np.median(depth))


def coc_map(depth, piece_mask, strength, max_frac=0.012):
    """Blur radius per pixel, in pixels. depth: 0 far .. 1 near (relative)."""
    h, w = depth.shape
    d = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 1.5)
    f = focus_depth(d, piece_mask)
    rng = max(float(np.percentile(d, 99) - np.percentile(d, 1)), 1e-3)
    coc = strength * max_frac * max(h, w) * np.clip(np.abs(d - f) / rng, 0, 1) ** 0.85
    # the piece and a margin around it stay in perfect focus
    keep = cv2.dilate((piece_mask > 0.02).astype(np.float32), np.ones((9, 9), np.uint8))
    keep = cv2.GaussianBlur(keep, (0, 0), 6)
    return (coc * (1 - np.clip(keep, 0, 1))).astype(np.float32)


def depth_of_field(img_lin, depth, piece_mask, strength=1.0):
    """Layered gather blur by circle of confusion. The piece region is exact."""
    depth = cv2.resize(depth, (img_lin.shape[1], img_lin.shape[0]), interpolation=cv2.INTER_LINEAR)
    coc = coc_map(depth, piece_mask, strength)
    rmax = float(coc.max())
    if rmax < 0.5:
        return img_lin.copy(), coc
    radii = np.linspace(0, rmax, LAYERS)
    layers = []
    for r in radii:
        k = _disc(r)
        layers.append(img_lin if k is None else cv2.filter2D(img_lin, -1, k, borderType=cv2.BORDER_REFLECT))
    t = coc / (radii[1] - radii[0])
    lo = np.clip(np.floor(t).astype(int), 0, LAYERS - 1)
    hi = np.clip(lo + 1, 0, LAYERS - 1)
    frac = (t - lo)[..., None].astype(np.float32)
    stack = np.stack(layers)
    yy, xx = np.indices(coc.shape)
    out = stack[lo, yy, xx] * (1 - frac) + stack[hi, yy, xx] * frac
    m = piece_mask[..., None]
    out = out * (1 - m) + img_lin * m  # exact pixels where the piece is
    return out.astype(np.float32), coc


def vignette(img_lin, strength=0.25):
    """Natural (cos^4-like) falloff toward the corners. Lighting only."""
    h, w = img_lin.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    r2 /= 2.0  # corner = 1
    v = (1.0 / (1.0 + strength * 2.2 * r2)) ** 2
    return (img_lin * v[..., None]).astype(np.float32)
