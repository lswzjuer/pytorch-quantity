#!/usr/bin/env python


"""
This module provide function to plot the speed control info from log csv file
"""

import sys
import math
import numpy as np
import tkFileDialog
from process import preprocess
from process import process
from process import get_start_index


class Plotter(object):
    """
    plot the speed info
    """

    def __init__(self, filename):
        """
        init the speed info
        """

        np.set_printoptions(precision=3)
        self.file = open('result.csv', 'a')
        self.file_one = open(filename + ".result", 'w')

    def process_data(self, filename):
        """
        load the file and preprocess th data
        """

        self.data = preprocess(filename)

        self.tablecmd, self.tablespeed, self.tableacc, self.speedsection, self.accsection, self.timesection = process(
            self.data)

    def save_data(self):
        """
        """
        for i in range(len(self.tablecmd)):
            for j in range(len(self.tablespeed[i])):
                self.file.write("%s, %s, %s\n" %
                                (self.tablecmd[i], self.tablespeed[i][j],
                                 self.tableacc[i][j]))
                self.file_one.write("%s, %s, %s\n" %
                                    (self.tablecmd[i], self.tablespeed[i][j],
                                     self.tableacc[i][j]))


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
    plotter = Plotter(file_path)
    plotter.process_data(file_path)
    plotter.save_data()
    print 'save result to:', file_path + ".result"


if __name__ == '__main__':
    main()
