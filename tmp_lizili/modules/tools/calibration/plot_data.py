#!/usr/bin/env python


"""
This module provide function to plot the speed control info from log csv file
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import tkFileDialog
from process import preprocess
from process import process
from process import get_start_index


class Plotter(object):
    """
    plot the speed info
    """

    def __init__(self):
        """
        init the speed info
        """

        np.set_printoptions(precision=3)
        self.file = open('temp_result.csv', 'a')

    def process_data(self, filename):
        """
        load the file and preprocess th data
        """

        self.data = preprocess(filename)

        self.tablecmd, self.tablespeed, self.tableacc, self.speedsection, self.accsection, self.timesection = process(
            self.data)

    def plot_result(self):
        """
        plot the desired data
        """
        fig, axarr = plt.subplots(2, 1, sharex=True)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)

        axarr[0].plot(
            self.data['time'], self.data['ctlbrake'], label='Brake CMD')
        axarr[0].plot(
            self.data['time'],
            self.data['brake_percentage'],
            label='Brake Output')
        axarr[0].plot(
            self.data['time'], self.data['ctlthrottle'], label='Throttle CMD')
        axarr[0].plot(
            self.data['time'],
            self.data['throttle_percentage'],
            label='Throttle Output')
        axarr[0].plot(
            self.data['time'],
            self.data['engine_rpm'] / 100,
            label='Engine RPM')
        axarr[0].legend(fontsize='medium')
        axarr[0].grid(True)
        axarr[0].set_title('Command')

        axarr[1].plot(
            self.data['time'],
            self.data['vehicle_speed'],
            label='Vehicle Speed')

        for i in range(len(self.timesection)):
            axarr[1].plot(
                self.timesection[i],
                self.speedsection[i],
                label='Speed Segment')
            axarr[1].plot(
                self.timesection[i], self.accsection[i], label='IMU Segment')

        axarr[1].legend(fontsize='medium')
        axarr[1].grid(True)
        axarr[1].set_title('Speed')

        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

        #plt.tight_layout(pad=0.20)
        fig.canvas.mpl_connect('key_press_event', self.press)
        plt.show()

    def press(self, event):
        """
        Keyboard events during plotting
        """
        if event.key == 'q' or event.key == 'Q':
            self.file.close()
            plt.close()

        if event.key == 'w' or event.key == 'W':
            for i in range(len(self.tablecmd)):
                for j in range(len(self.tablespeed[i])):
                    self.file.write("%s, %s, %s\n" %
                                    (self.tablecmd[i], self.tablespeed[i][j],
                                     self.tableacc[i][j]))
            print "Done writing results"


def main():
    """
    demo
    """
    if len(sys.argv) == 2:
        # get the latest file
        file_path = sys.argv[1]
    else:
        file_path = tkFileDialog.askopenfilename(
            initialdir="/home/caros/.ros",
            filetypes=(("csv files", ".csv"), ("all files", "*.*")))
    print 'File path:', file_path
    plotter = Plotter()
    plotter.process_data(file_path)
    print 'Done reading the file.'
    plotter.plot_result()


if __name__ == '__main__':
    main()
