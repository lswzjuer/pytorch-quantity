#! /usr/bin/python
# -*- coding: UTF-8 -*-

from modules.msgs.hdmap.proto.lanemarker_pb2 import Lanemarker
from easy_plot import EasyPlot


class HdmapRender:
    def __init__(self):
        self.easy_plot = EasyPlot()

    def ploter(self):
        return self.easy_plot

    def draw_polygon(self, polygon, **kwargs):
        x = []
        y = []
        for point in polygon.points:
            x.append(point.x)
            y.append(point.y)
        x.append(polygon.points[0].x)
        y.append(polygon.points[0].y)
        self.easy_plot.draw_line(x=x, y=y, **kwargs)

    def draw_section(self, section, **kwargs):
        # draw_polygon first
        self.draw_polygon(section.polygon, **kwargs)
        # then draw_laneker
        for index in range(len(section.lanemarkers)):
            if index == 0 or index == len(section.lanemarkers) - 1:
                continue
            self.draw_lanemarker(section.lanemarkers[index])

    def draw_connection(self, connection, **kwargs):
        self.draw_polygon(connection.polygon, **kwargs)

    def draw_lanemarker(self, lanemarker):
        color = ''
        linestyle = ''
        if lanemarker.type == Lanemarker.LANEMARKER_UNKNOWN:
            color = 'k'  #means black
            linestyle = '-.'
        elif lanemarker.type == Lanemarker.LANEMARKER_DASHED_WHITE:
            color = 'c'
            linestyle = 'dashed'
        elif lanemarker.type == Lanemarker.LANEMARKER_SOLID_WHITE:
            color = 'c'
            linestyle = 'solid'
        elif lanemarker.type == Lanemarker.LANEMARKER_SOLID_YELLOW:
            color = 'y'
            linestyle = 'solid'
        elif lanemarker.type == Lanemarker.LANEMARKER_DASHED_YELLOW:
            color = 'y'
            linestyle = 'dashed'
        elif lanemarker.type == Lanemarker.LANEMARKER_VIRTUAL:
            color = 'k'  #means black
            linestyle = '-.'
        self.draw_curve(lanemarker.curve, c=color, ls=linestyle)

    def draw_curve(self, curve, **kwargs):
        for index in range(len(curve.points)):
            if index + 1 >= len(curve.points):
                break
            self.draw_line(curve.points[index], curve.points[index + 1],
                           **kwargs)

    def draw_point(self, point, **kwargs):
        self.easy_plot.draw_point([point.x, point.y], **kwargs)

    def draw_line(self, from_pt, to_pt, **kwargs):
        self.easy_plot.draw_line(
            x=[from_pt.x, to_pt.x], y=[from_pt.y, to_pt.y], **kwargs)
