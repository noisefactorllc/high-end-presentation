import numpy as np
import pytest

from conftest import place
from hep import locate


def test_recovers_known_homography(piece, scene, quad):
    composed, H = place(piece, scene, quad)
    loc = locate.find_piece(piece, composed)
    assert loc.ok, loc.reason
    assert np.abs(loc.quad - np.array(quad)).max() < 2.0


def test_missing_piece_reports_reason(piece, scene):
    loc = locate.find_piece(piece, scene)
    assert not loc.ok and loc.reason


def test_located_json_roundtrip(piece, scene, quad):
    composed, _ = place(piece, scene, quad)
    loc = locate.find_piece(piece, composed)
    back = locate.Located.from_json(loc.to_json())
    assert np.allclose(back.H, loc.H) and back.ok


def test_convexity():
    assert locate.is_convex([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert not locate.is_convex([[0, 0], [10, 10], [10, 0], [0, 10]])


def _project(w, h, yaw_deg, pitch_deg, f=2800.0, dist=3.0, size=(1600, 2000)):
    """Image quad (TL, TR, BR, BL) of a w x h rectangle seen by a pinhole camera."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    pts = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    cam = (Rx @ Ry @ pts.T).T + [0.1, -0.05, dist]
    uv = cam[:, :2] / cam[:, 2:] * f + np.array(size) / 2
    return uv


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (4, 0), (25, 0), (-35, 10), (15, -20)])
def test_rectangle_aspect_recovers_true_proportions(yaw, pitch):
    quad = _project(1.134, 1.0, yaw, pitch)
    est = locate.rectangle_aspect(quad, (1600, 2000))
    assert abs(est / 1.134 - 1) < 0.02


def test_rectangle_aspect_detects_squeeze():
    quad = _project(1.134, 1.0, 0, 0)
    c = quad.mean(0)
    squeezed = (quad - c) * [0.91, 1.0] + c
    assert abs(locate.rectangle_aspect(squeezed, (1600, 2000)) / 1.134 - 1) > 0.06
