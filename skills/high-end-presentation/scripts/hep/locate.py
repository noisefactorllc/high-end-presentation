"""Find the piece inside a generated scene with SIFT + RANSAC."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import color, media


@dataclass
class Located:
    ok: bool
    H: np.ndarray = field(default_factory=lambda: np.eye(3))  # piece px -> scene px
    quad: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))  # TL, TR, BR, BL in scene px
    inliers: int = 0
    matches: int = 0
    reason: str = ""

    def to_json(self):
        return {"ok": self.ok, "H": np.asarray(self.H).tolist(), "quad": np.asarray(self.quad).round(2).tolist(),
                "inliers": self.inliers, "matches": self.matches, "reason": self.reason}

    @classmethod
    def from_json(cls, d):
        return cls(d["ok"], np.array(d["H"], np.float64), np.array(d["quad"], np.float64),
                   d["inliers"], d["matches"], d["reason"])


def _gray8(img, max_side):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = media.resize(img, round(w * s), round(h * s)) if s < 1 else img
    g = color.luma(small)
    g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g), s


def corners(w, h):
    return np.float64([[0, 0], [w, 0], [w, h], [0, h]])


def is_convex(quad):
    q = np.asarray(quad, np.float64)
    cross = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        u, v = b - a, c - b
        cross.append(u[0] * v[1] - u[1] * v[0])
    cross = np.array(cross)
    return bool(np.all(cross > 0) or np.all(cross < 0))


def quad_area(quad):
    q = np.asarray(quad, np.float64)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def find_piece(piece, scene, piece_side=1600, scene_side=2600, min_inliers=25):
    gp, sp = _gray8(piece, piece_side)
    gs, ss = _gray8(scene, scene_side)
    sift = cv2.SIFT_create(nfeatures=12000)
    kp1, d1 = sift.detectAndCompute(gp, None)
    kp2, d2 = sift.detectAndCompute(gs, None)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return Located(False, reason="no-features")
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    good = []
    for pair in matcher.knnMatch(d1, d2, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < min_inliers:
        return Located(False, matches=len(good), reason="too-few-matches")
    src = np.float32([kp1[m.queryIdx].pt for m in good]) / sp
    dst = np.float32([kp2[m.trainIdx].pt for m in good]) / ss
    H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0 / ss, maxIters=10000, confidence=0.999)
    if H is None:
        return Located(False, matches=len(good), reason="no-homography")
    n_in = int(inl.sum())
    h, w = piece.shape[:2]
    quad = cv2.perspectiveTransform(corners(w, h)[None], H)[0]
    if n_in < min_inliers:
        return Located(False, H, quad, n_in, len(good), "too-few-inliers")
    if not is_convex(quad):
        return Located(False, H, quad, n_in, len(good), "non-convex")
    return Located(True, H, quad, n_in, len(good), "")


def flatten(scene, H, size):
    """Warp the scene back into the piece's frame. size = (width, height) of output."""
    w, h = size
    return cv2.warpPerspective(scene, np.linalg.inv(H), (int(w), int(h)), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def scaled_H(H, piece_scale):
    """H for a piece resized by piece_scale (piece px' = piece px * s)."""
    S = np.diag([1.0 / piece_scale, 1.0 / piece_scale, 1.0])
    return H @ S
