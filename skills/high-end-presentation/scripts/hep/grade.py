"""Tone map, palette split-tone, grain. Output is sRGB."""
import numpy as np
import cv2

from . import color


def tone_map(img_lin):
    """Hue-preserving extended Reinhard on luminance. Identity for images that
    are already within [0, 1]; compresses trail highlights above 1."""
    L = color.luma(img_lin)
    Lw = max(1.0, float(np.percentile(L, 99.95)))
    if Lw <= 1.0 + 1e-6:
        out = img_lin
    else:
        Lo = L * (1 + L / (Lw * Lw)) / (1 + L)
        # keep midtones where they were: only highlights move much
        blend = np.clip(L / 0.8, 0, 1) ** 2
        Lt = L * (1 - blend) + Lo * blend * (0.8 * (1 + 0.8) / (0.8 * (1 + 0.8 / (Lw * Lw))))
        out = img_lin * (Lt / np.maximum(L, 1e-6))[..., None]
    return np.clip(color.linear_to_srgb(out), 0, 1)


def s_curve(srgb, amount=0.08):
    x = np.clip(srgb, 0, 1)
    return x + amount * (x * x * (3 - 2 * x) - x)


def _tone_targets(palette):
    """(shadow_ab, highlight_ab) unit-ish directions from the palette."""
    labs = np.array([p["lab"] for p in palette], np.float32)
    weights = np.array([p["weight"] for p in palette], np.float32)
    chroma = np.hypot(labs[:, 1], labs[:, 2])
    usable = chroma > 6
    if not usable.any():
        return np.zeros(2, np.float32), np.zeros(2, np.float32)
    labs, weights, chroma = labs[usable], weights[usable], chroma[usable]
    score_dark = weights * chroma * (100 - labs[:, 0])
    score_light = weights * chroma * labs[:, 0]
    d = labs[np.argmax(score_dark), 1:] / chroma[np.argmax(score_dark)]
    l = labs[np.argmax(score_light), 1:] / chroma[np.argmax(score_light)]
    return d.astype(np.float32), l.astype(np.float32)


def split_tone(srgb, palette, strength=0.15, chroma=14.0):
    lab = color.rgb_to_lab(srgb)
    shadow, high = _tone_targets(palette)
    Ln = np.clip(lab[..., 0] / 100.0, 0, 1)
    ws = ((1 - Ln) ** 2)[..., None]
    wh = (Ln ** 2)[..., None]
    lab[..., 1:] += strength * chroma * (ws * shadow + wh * high)
    return color.lab_to_rgb(lab)


def grain(srgb, amount=0.012, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, srgb.shape[:2]).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.6)
    n /= n.std() + 1e-6
    mid = 4 * srgb.mean(-1) * (1 - srgb.mean(-1))  # grain lives in the midtones
    return np.clip(srgb + amount * n[..., None] * mid[..., None], 0, 1)


def apply(img_lin, palette, piece_mask, *, tone=0.15, contrast=0.08, grain_amount=0.012,
          piece_strength=0.5, seed=0):
    base = tone_map(img_lin)
    graded = grain(split_tone(s_curve(base, contrast), palette, tone), grain_amount, seed)
    piece_view = base + piece_strength * (graded - base)
    m = piece_mask[..., None]
    return np.clip(graded * (1 - m) + piece_view * m, 0, 1).astype(np.float32)
