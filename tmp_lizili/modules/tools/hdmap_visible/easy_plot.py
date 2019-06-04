#! /usr/bin/python
# -*- coding: UTF-8 -*-

import time
import matplotlib as mpl
# mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


class EasyPlot:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def ion(self):
        plt.ion()

    def add_subplot(self):
        self.ax = self.fig.add_subplot(111)
        return self.ax

    def figure(self):
        self.fig = plt.figure()
        return self.fig

    def draw_axis(self,
                  xmin=-10,
                  xmax=10,
                  ymin=-10,
                  ymax=10,
                  xlabel='X',
                  ylabel=''):
        plt.axis([xmin, xmax, ymin,
                  ymax])  #other mode like : scaled, autod, off
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def draw_point(self, xy=[0, 0], **kwargs):
        plt.plot(xy[0], xy[1], **kwargs)

    def draw_line(self, x=[], y=[], **kwargs):
        # eg:c = color,marker = mker,lw = linewidth,ls = linestyle,
        # alpha = alpha，label=label
        plt.plot(x, y, figure=self.fig, **kwargs)

    def pause(self, second):
        plt.pause(second)

    def draw_fig(self):
        plt.draw()

    def show_fig(self):
        plt.show()

    def clear_fig(self):
        plt.clf()

    def close_fig(self):
        plt.close()
