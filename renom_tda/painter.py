# -*- coding: utf-8 -*.t-
from __future__ import print_function
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass
import functools
import colorsys


class Painter(with_metaclass(ABCMeta, object)):
    def is_type(self, t):
        if t == self.color_type:
            return True
        return False

    def _check_data(func):
        @functools.wraps(func)
        def wrapper(*args):
            if args[1] is None:
                raise ValueError("Input data is None.")
            return func(*args)
        return wrapper

    @abstractmethod
    def paint(self):
        pass


class RGBPainter(Painter):
    def __init__(self):
        self.color_type = "rgb"

    @Painter._check_data
    def paint(self, v):
        c = colorsys.hsv_to_rgb((1 - v) * 240 / 360, 1.0, 0.7)
        return "#%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


class GrayPainter(Painter):
    def __init__(self):
        self.color_type = "gray"

    @Painter._check_data
    def paint(self, v):
        v = 1 - v
        return "#%02x%02x%02x" % (int((v + 0.1) * 200), int((v + 0.1) * 200), int((v + 0.1) * 200))


class PainterResolver(object):
    def __init__(self):
        self.painter_list = [RGBPainter(), GrayPainter()]

    def resolve(self, color_type):
        for p in self.painter_list:
            if p.is_type(color_type):
                return p
