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


def _edge_angle(a, b):
    return np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))


def _near_parallel(quad, tol_deg=4.0):
    """True when both pairs of opposite edges are nearly parallel: the view is
    close to frontal, and the focal length cannot be recovered reliably."""
    tl, tr, br, bl = (np.asarray(q, np.float64) for q in quad)
    def diff(a, b):
        d = abs(a - b) % 180
        return min(d, 180 - d)
    return (diff(_edge_angle(tl, tr), _edge_angle(bl, br)) < tol_deg and
            diff(_edge_angle(tl, bl), _edge_angle(tr, br)) < tol_deg)


def rectangle_aspect(quad, image_size):
    """Physical width/height of the rectangle whose image is quad (TL, TR, BR,
    BL), assuming square pixels and the principal point at the image center
    (Zhang & He, rectangle rectification). Handles frontal views too."""
    w, h = image_size
    c = np.array([w / 2.0, h / 2.0])
    tl, tr, br, bl = (np.asarray(q, np.float64) - c for q in quad)
    m1, m2, m3, m4 = (np.array([p[0], p[1], 1.0]) for p in (tl, tr, bl, br))
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1
    scale = max(np.abs(n2[:2]).max(), np.abs(n3[:2]).max())
    den = n2[2] * n3[2]
    f2 = -(n2[0] * n3[0] + n2[1] * n3[1]) / den if abs(den) > 1e-9 * scale * scale else -1.0
    lo, hi = (0.5 * max(w, h)) ** 2, (4.0 * max(w, h)) ** 2
    if not np.isfinite(f2) or not lo <= f2 <= hi or _near_parallel(quad):
        # One or both side pairs are (nearly) parallel in the image, so the
        # focal length is not observable. Use a normal-lens prior (about a
        # 35-50 mm equivalent); frontal views do not depend on it.
        f2 = (1.4 * max(w, h)) ** 2
    Ainv2 = np.diag([1.0 / f2, 1.0 / f2, 1.0])
    return float(np.sqrt((n2 @ Ainv2 @ n2) / (n3 @ Ainv2 @ n3)))


def order_quad(pts):
    """Order four points as TL, TR, BR, BL."""
    pts = np.asarray(pts, np.float64).reshape(4, 2)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # clockwise in image coordinates, starting near -pi (left)
    start = int(np.argmin(pts.sum(1)))  # top-left has the smallest x + y
    return np.roll(pts, -start, axis=0)


def candidate_quads(scene, min_frac=0.005, max_frac=0.6, limit=40):
    """Convex quadrilaterals in the scene (screens, frames, mats), largest first."""
    h, w = scene.shape[:2]
    s = min(1.0, 1400 / max(h, w))
    g = (np.clip(color.luma(media.resize(scene, round(w * s), round(h * s))), 0, 1) * 255).astype(np.uint8)
    found = []
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    masks = [edges] + [cv2.threshold(g, t, 255, cv2.THRESH_BINARY)[1] for t in (60, 110, 160, 210)]
    area_img = g.shape[0] * g.shape[1]
    for m in masks:
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            a = cv2.contourArea(c)
            if not min_frac * area_img <= a <= max_frac * area_img:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                found.append((a, order_quad(approx[:, 0, :] / s)))
    found.sort(key=lambda t: -t[0])
    out = []
    for a, q in found:
        if all(np.abs(q - o).max() > 6 for o in out):
            out.append(q)
        if len(out) >= limit:
            break
    return out


def _bandpass(g):
    g = g.astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 2.0) - cv2.GaussianBlur(g, (0, 0), 8.0)


def refine(piece, scene, H, side=512):
    """Refine H (piece px -> scene px) by ECC on band-pass luma, which ignores
    the scene's tone changes. Returns (H, ok)."""
    ph, pw = piece.shape[:2]
    s = side / max(ph, pw)
    tmpl = _bandpass(color.luma(media.resize(piece, round(pw * s), round(ph * s))))
    # scene at a matching scale around the piece
    quad = cv2.perspectiveTransform(corners(pw, ph)[None], H)[0]
    foot = np.sqrt(quad_area(quad) / (pw * ph)) * s
    ss = min(1.0, 1.0 / max(foot, 1e-6))
    sh, sw = scene.shape[:2]
    img = _bandpass(color.luma(media.resize(scene, round(sw * ss), round(sh * ss))))
    W = (np.diag([ss, ss, 1.0]) @ H @ np.diag([1 / s, 1 / s, 1.0])).astype(np.float32)
    try:
        _, W = cv2.findTransformECC(tmpl, img, W, cv2.MOTION_HOMOGRAPHY,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6), None, 3)
    except cv2.error:
        return H, False
    Hn = np.diag([1 / ss, 1 / ss, 1.0]) @ W.astype(np.float64) @ np.diag([s, s, 1.0])
    return Hn / Hn[2, 2], True


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
