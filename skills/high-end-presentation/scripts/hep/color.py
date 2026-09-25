"""Color-space helpers. All arrays are float32; sRGB values are in [0, 1]."""
import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
# sRGB (D65) <-> XYZ
_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], np.float32)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ).astype(np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], np.float32)


def srgb_to_linear(a):
    a = np.asarray(a, np.float32)
    return np.where(a <= 0.04045, a / 12.92, ((np.maximum(a, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(a):
    a = np.clip(np.asarray(a, np.float32), 0, None)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055).astype(np.float32)


def luma(a):
    """Relative luminance of linear RGB (or luma of sRGB, if given sRGB)."""
    return np.asarray(a, np.float32) @ _LUMA


def rgb_to_lab(srgb):
    xyz = srgb_to_linear(srgb) @ _RGB2XYZ.T / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    A = 500 * (f[..., 0] - f[..., 1])
    B = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, A, B], -1).astype(np.float32)


def lab_to_rgb(lab):
    lab = np.asarray(lab, np.float32)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WHITE
    return np.clip(linear_to_srgb(xyz @ _XYZ2RGB.T), 0, 1)


def to_hex(srgb):
    r, g, b = (int(round(float(c) * 255)) for c in np.clip(srgb, 0, 1))
    return f"#{r:02x}{g:02x}{b:02x}"
