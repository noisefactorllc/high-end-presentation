import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/high-end-presentation/scripts"))


def textured_piece(h=360, w=480, seed=1):
    """A deterministic, feature-rich synthetic artwork (sRGB float)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img[..., 0] = 0.5 + 0.4 * np.sin(xx / 17.0 + seed)
    img[..., 1] = 0.5 + 0.4 * np.sin(yy / 23.0 + xx / 41.0)
    img[..., 2] = 0.5 + 0.4 * np.cos((xx + yy) / 29.0)
    for _ in range(220):
        c = rng.random(3).astype(np.float32)
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(4, 26))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, c.tolist(), -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + r // 2 + 3), c.tolist(), -1)
    return np.clip(img, 0, 1)


def plain_scene(h=900, w=720, seed=5):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.55 + 0.1 * (yy / h)
    img = np.stack([base, base * 0.97, base * 0.92], -1)
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def place(piece, scene, quad):
    """Warp piece into scene at quad (TL, TR, BR, BL). Returns scene, H."""
    h, w = piece.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, np.float32(quad))
    warped = cv2.warpPerspective(piece, H, (scene.shape[1], scene.shape[0]), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), H, (scene.shape[1], scene.shape[0]))
    out = scene * (1 - mask[..., None]) + warped * mask[..., None]
    return out.astype(np.float32), H


@pytest.fixture
def piece():
    return textured_piece()


@pytest.fixture
def scene():
    return plain_scene()


def project_quad(aspect, yaw_deg, pitch_deg, width_px, center, f=1400.0, size=(720, 900)):
    """Image quad (TL, TR, BR, BL) of a rectangle with this aspect seen by a
    pinhole camera, scaled so its top edge is about width_px wide."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    pts = np.array([[-aspect / 2, -0.5, 0], [aspect / 2, -0.5, 0], [aspect / 2, 0.5, 0], [-aspect / 2, 0.5, 0]])
    dist = f * aspect / width_px
    cam = (Rx @ Ry @ pts.T).T + [(center[0] - size[0] / 2) * dist / f, (center[1] - size[1] / 2) * dist / f, dist]
    return (cam[:, :2] / cam[:, 2:] * f + np.array(size) / 2).tolist()


# a 480x360 piece seen slightly from the left and above
QUAD = project_quad(480 / 360, 12, -6, 380, (370, 380))


@pytest.fixture
def quad():
    return QUAD
