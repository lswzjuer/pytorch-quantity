#!/usr/bin/env python


"""
Time Value Item
"""
import numpy
from matplotlib import lines


class Item(object):
    """
    Specific item to plot
    """

    def __init__(self, ax, title, xlabel, ylabel, ymin, ymax):
        self.ax = ax

        self.title = title
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel(ylabel, fontsize=10)
        self.ymin = ymin
        self.ymax = ymax
        self.ax.set_ylim([ymin, ymax])

        self.lines = []

        self.cartimehist = []
        self.carvaluehist = []
        self.targettime = []
        self.targethist = []

        self.targethistidx = -1
        self.histidx = -1

        self.prev_auto = False

        self.planningavailable = False

    def reset(self):
        """
        Reset
        """
        del self.lines[:]

        del self.cartimehist[:]
        del self.carvaluehist[:]
        del self.targettime[:]
        del self.targethist[:]

        self.ax.cla()

        self.ax.set_ylim([self.ymin, self.ymax])
        self.targethistidx = -1
        self.histidx = -1

        self.prev_auto = False

        self.planningavailable = False

    def new_planning(self, time, values):
        """
        new planning
        """
        self.planningtime = time
        self.planningvalues = values

        if self.planningavailable == False:
            self.ax.set_xlim([time[0] - 1, time[-1] + 10])
            self.current_line = lines.Line2D(time, values, color='red', lw=1.5)
            self.ax.add_line(self.current_line)

        else:
            self.current_line.set_data(time, values)

            xmin, xmax = self.ax.get_xlim()
            if (time[-1] >= (xmax - 1)):
                self.ax.set_xlim([time[0] - 1, time[-1] + 10])

        self.planningavailable = True

    def new_carstatus(self, time, value, autodriving):
        """
        new carstatus
        """
        if autodriving and not self.prev_auto:
            self.starttime = time
            self.endtime = time + 50
            self.ax.axvspan(self.starttime, self.endtime, fc='0.1', alpha=0.3)
        elif autodriving and time >= (self.endtime - 20):
            self.endtime = time + 50
            self.ax.patches[-1].remove()
            self.ax.axvspan(self.starttime, self.endtime, fc='0.1', alpha=0.3)
        elif not autodriving and self.prev_auto:
            self.endtime = time
            self.ax.patches[-1].remove()
            self.ax.axvspan(self.starttime, self.endtime, fc='0.1', alpha=0.3)

        self.prev_auto = autodriving

        self.cartimehist.append(time)
        self.carvaluehist.append(value)

        if self.planningavailable:
            target = numpy.interp(time, self.planningtime, self.planningvalues)
            self.targettime.append(time)
            self.targethist.append(target)

            if self.targethistidx == -1:
                self.ax.plot(
                    self.targettime, self.targethist, color='green', lw=1.5)
                self.targethistidx = len(self.ax.lines) - 1
            else:
                self.ax.lines[self.targethistidx].set_data(
                    self.targettime, self.targethist)

        if self.histidx == -1:
            self.ax.plot(self.cartimehist, self.carvaluehist, color='blue')
            self.histidx = len(self.ax.lines) - 1

        else:
            self.ax.lines[self.histidx].set_data(self.cartimehist,
                                                 self.carvaluehist)

    def draw_lines(self):
        """
        plot lines
        """
        for polygon in self.ax.patches:
            self.ax.draw_artist(polygon)

        if self.planningavailable:
            self.ax.draw_artist(self.current_line)

        if self.targethistidx != -1:
            self.ax.draw_artist(self.ax.lines[self.targethistidx])

        if self.histidx != -1:
            self.ax.draw_artist(self.ax.lines[self.histidx])
