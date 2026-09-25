import numpy as np

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
