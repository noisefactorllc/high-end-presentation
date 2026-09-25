"""Display treatments, applied by the skill and never by a model: the artist's
own pixels, shown the way a physical display would show them."""
import cv2
import numpy as np

from . import color, media

SCREEN_PX = 768  # long side of a rendered screen image


def screen_canvas(piece, screen_aspect, side=SCREEN_PX):
    """The piece centred on a black screen of this aspect, at its true
    proportions (pillarboxed or letterboxed, never cropped or stretched)."""
    if screen_aspect >= 1:
        W, H = side, max(8, round(side / screen_aspect))
    else:
        W, H = max(8, round(side * screen_aspect)), side
    ph, pw = piece.shape[:2]
    s = min(W / pw, H / ph)
    fw, fh = max(1, round(pw * s)), max(1, round(ph * s))
    canvas = np.zeros((H, W, 3), np.float32)
    x0, y0 = (W - fw) // 2, (H - fh) // 2
    canvas[y0:y0 + fh, x0:x0 + fw] = media.resize(piece, fw, fh, interpolation=cv2.INTER_AREA)
    return canvas


def crt(screen_srgb, curvature=0.08, scanlines=0.3, glow=0.22, corner_power=7.0):
    """Picture-tube rendering of a screen image: barrel curvature, scanlines,
    phosphor glow, and the rounded, darkening edge of the tube.
    Returns (image linear, alpha)."""
    lin = color.srgb_to_linear(screen_srgb)
    h, w = lin.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx + 0.5) / w * 2 - 1
    v = (yy + 0.5) / h * 2 - 1
    r2 = u * u + v * v
    su = u * (1 + curvature * r2)
    sv = v * (1 + curvature * r2)
    mapx = (su + 1) / 2 * w - 0.5
    mapy = (sv + 1) / 2 * h - 0.5
    img = cv2.remap(lin, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lines = max(60.0, h / 3.0)
    img *= (1 - scanlines * (0.5 + 0.5 * np.cos(2 * np.pi * (yy + 0.5) * lines / h)))[..., None]
    img += glow * cv2.GaussianBlur(img, (0, 0), max(1.0, 0.012 * w))
    img *= (1 - 0.35 * np.clip(r2 / 2, 0, 1))[..., None]
    shape = np.abs(su) ** corner_power + np.abs(sv) ** corner_power
    alpha = np.clip((1.0 - shape) / 0.08, 0, 1).astype(np.float32)
    return img.astype(np.float32), alpha


TREATMENTS = {"flat", "crt"}
