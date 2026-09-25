import numpy as np

from conftest import textured_piece
from hep import color, display


def test_screen_canvas_pillarboxes_without_cropping():
    piece = textured_piece(200, 200, seed=2)
    c = display.screen_canvas(piece, 4 / 3, side=400)
    assert c.shape[:2] == (300, 400)
    assert c[:, :40].max() == 0 and c[:, -40:].max() == 0  # dark bars at the sides
    assert c[:, 60:340].mean() > 0.2                       # the whole piece, centred


def test_crt_has_rounded_edges_and_scanlines():
    img, alpha = display.crt(np.full((300, 400, 3), 0.6, np.float32))
    assert alpha[150, 200] == 1.0 and alpha[0, 0] == 0.0
    column = color.luma(img)[120:180, 200]
    assert column.max() - column.min() > 0.05  # scanlines modulate the rows
