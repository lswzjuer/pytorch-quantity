#!/usr/bin/env python


"""
S T Item
"""
from matplotlib import lines


class Stitem(object):
    """
    Specific item to plot
    """

    def __init__(self, ax, lines2display, title, xlabel, ylabel):
        self.ax = ax
        self.lines2display = lines2display

        self.title = title
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel(ylabel, fontsize=10)

        self.lines = []

        self.planningavailable = False

    def reset(self):
        """
        Reset
        """
        del self.lines[:]

        self.ax.cla()

        self.ax.set_xlim([-0.1, 0.1])
        self.ax.set_ylim([-0.1, 0.1])

        self.planningavailable = False

    def new_planning(self, time, values, maxtime, maxvalue):
        """
        new planning
        """
        if self.planningavailable == False:
            self.ax.set_xlim([0, maxtime + 1])
            self.ax.set_ylim([0, maxvalue + 10])
            self.ymax = maxvalue
            self.tmax = maxtime

        else:
            self.current_line.set_color('cyan')
            self.current_line.set_linestyle('dashed')
            self.current_line.set_linewidth(1.5)
            self.lines.append(self.current_line)

            xmin, xmax = self.ax.get_xlim()
            if maxtime > xmax:
                self.ax.set_xlim([0, maxtime])

            ymin, ymax = self.ax.get_ylim()
            if maxvalue > ymax:
                self.ax.set_ylim([0, maxvalue + 10])

        self.current_line = lines.Line2D(time, values, color='red', lw=1.5)
        self.ax.add_line(self.current_line)
        self.planningavailable = True

    def draw_lines(self):
        """
        plot lines
        """
        for polygon in self.ax.patches:
            self.ax.draw_artist(polygon)

        for i in range(
                max(0, len(self.lines) - self.lines2display), len(self.lines)):
            self.ax.draw_artist(self.lines[i])

        if self.planningavailable:
            self.ax.draw_artist(self.current_line)
