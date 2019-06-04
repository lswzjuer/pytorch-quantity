#!/usr/bin/env python


"""
This module provide function to plot the speed control info from log csv file
"""

import numpy as np
import warnings
import math
import scipy.signal as signal
warnings.simplefilter('ignore', np.RankWarning)

SPEED_INTERVAL = 0.2
SPEED_DELAY = 130  #Speed report delay relative to IMU information


def preprocess(filename):
    data = np.genfromtxt(filename, delimiter=',', names=True)
    data = data[np.where(data['io'] == 0)[0]]
    data = data[np.argsort(data['time'])]
    data['time'] = data['time'] - data['time'][get_start_index(data)]

    b, a = signal.butter(6, 0.05, 'low')
    data['imu'] = signal.filtfilt(b, a, data['imu'])

    data['imu'] = np.append(data['imu'][-SPEED_DELAY / 10:],
                            data['imu'][0:-SPEED_DELAY / 10])
    return data


def get_start_index(data):
    if np.all(data['vehicle_speed'] == 0):
        return 0

    start_ind = np.where(data['brake_percentage'] == 40)[0][0]

    ind = start_ind
    while ind < len(data):
        if data['brake_percentage'][
                ind] == 40:  #or data['vehicle_speed'][ind] == 0.0:
            ind = ind + 1
        else:
            break
    return ind


def process(data):
    """
    process data
    """
    np.set_printoptions(precision=3)

    if np.all(data['vehicle_speed'] == 0):
        print "All Speed = 0"
        return [], [], [], [], [], []

    start_index = get_start_index(data)

    #print "Start index: ", start_index
    data = data[start_index:]
    data['time'] = data['time'] - data['time'][0]

    transition = np.where(
        np.logical_or(
            np.diff(data['ctlbrake']) != 0, np.diff(data['ctlthrottle']) != 0))[
                0]
    transition = np.insert(np.append(transition, len(data) - 1), 0, 0)
    #print "Transition indexes: ", transition

    speedsegments = []
    timesegments = []
    accsegments = []
    tablespeed = []
    tableacc = []
    tablecmd = []

    for i in range(len(transition) - 1):
        #print "process transition index:", data['time'][transition[i]], ":", data['time'][transition[i + 1]]
        speedsection = data['vehicle_speed'][transition[i]:transition[i +
                                                                      1] + 1]
        timesection = data['time'][transition[i]:transition[i + 1] + 1]
        brake = data['ctlbrake'][transition[i] + 1]
        throttle = data['ctlthrottle'][transition[i] + 1]
        imusection = data['imu'][transition[i]:transition[i + 1] + 1]
        if brake == 0 and throttle == 0:
            continue
        #print "Brake CMD: ", brake, " Throttle CMD: ", throttle
        firstindex = 0

        while speedsection[firstindex] == 0:
            firstindex = firstindex + 1
        firstindex = max(firstindex - 2, 0)
        speedsection = speedsection[firstindex:]
        timesection = timesection[firstindex:]
        imusection = imusection[firstindex:]

        if speedsection[0] < speedsection[-1]:
            is_increase = True
            lastindex = np.argmax(speedsection)
        else:
            is_increase = False
            lastindex = np.argmin(speedsection)

        speedsection = speedsection[0:lastindex + 1]
        timesection = timesection[0:lastindex + 1]
        imusection = imusection[0:lastindex + 1]

        speedmin = np.min(speedsection)
        speedmax = np.max(speedsection)
        speedrange = np.arange(
            max(0, round(speedmin / SPEED_INTERVAL) * SPEED_INTERVAL),
            min(speedmax, 10.01), SPEED_INTERVAL)
        #print "Speed min, max", speedmin, speedmax, is_increase, firstindex, lastindex, speedsection[-1]
        accvalue = []
        for value in speedrange:
            val_ind = 0
            if is_increase:
                while val_ind < len(
                        speedsection) - 1 and value > speedsection[val_ind]:
                    val_ind = val_ind + 1
            else:
                while val_ind < len(
                        speedsection) - 1 and value < speedsection[val_ind]:
                    val_ind = val_ind + 1
            if val_ind == 0:
                imu_value = imusection[val_ind]
            else:
                slope = (imusection[val_ind] - imusection[val_ind - 1]) / (
                    speedsection[val_ind] - speedsection[val_ind - 1])
                imu_value = imusection[val_ind - 1] + slope * (
                    value - speedsection[val_ind - 1])
            accvalue.append(imu_value)

        if brake == 0:
            cmd = throttle
        else:
            cmd = -brake
        #print "Overall CMD: ", cmd
        #print "Time: ", timesection
        #print "Speed: ", speedrange
        #print "Acc: ", accvalue
        #print cmd
        tablecmd.append(cmd)
        tablespeed.append(speedrange)
        tableacc.append(accvalue)

        speedsegments.append(speedsection)
        accsegments.append(imusection)
        timesegments.append(timesection)

    return tablecmd, tablespeed, tableacc, speedsegments, accsegments, timesegments
