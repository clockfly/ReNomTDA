import pytest
from renom_tda.api.painter import RGBPainter, GrayPainter, PainterResolver


def test_rgb_painter_min():
    painter = RGBPainter()
    c = painter.paint(0)
    test_c = "#0000b2"

    assert c == test_c


def test_rgb_painter_max():
    painter = RGBPainter()
    c = painter.paint(1)
    test_c = "#b20000"

    assert c == test_c


def test_rgb_painter_none_input():
    painter = RGBPainter()
    with pytest.raises(Exception):
        painter.paint(None)


def test_gray_painter_min():
    painter = GrayPainter()
    c = painter.paint(0)
    test_c = "#dcdcdc"

    assert c == test_c


def test_gray_painter_max():
    painter = GrayPainter()
    c = painter.paint(1)
    test_c = "#141414"

    assert c == test_c


def test_gray_painter_none_input():
    painter = GrayPainter()
    with pytest.raises(Exception):
        painter.paint(None)


def test_resolver_rgb():
    resolver = PainterResolver()
    painter = resolver.resolve("rgb")
    assert isinstance(painter, RGBPainter)


def test_resolver_gray():
    resolver = PainterResolver()
    painter = resolver.resolve("gray")
    assert isinstance(painter, GrayPainter)
