#!/usr/bin/env python



import matplotlib.pyplot as plt
import math
import sys
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import cm as cmx

markers = [
    "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "+", "x",
    "d", "|", "_"
]

if len(sys.argv) < 2:
    print "usage: python plot_results.py result.csv"
    sys.exit()

fn = sys.argv[1]

speed_table = {}
f = open(fn, 'r')
for line in f:
    items = line.split(',')
    cmd = round(float(items[0]))
    speed = float(items[1])
    acc = round(float(items[2]), 2)
    if speed in speed_table:
        cmd_table = speed_table[speed]
        if cmd in cmd_table:
            cmd_table[cmd].append(acc)
        else:
            cmd_table[cmd] = [acc]
    else:
        cmd_table = {}
        cmd_table[cmd] = [acc]
        speed_table[speed] = cmd_table
f.close()

for speed, cmd_dict in speed_table.items():
    speed_list = []
    acc_list = []
    for cmd, accs in cmd_dict.items():
        for acc in accs:
            speed_list.append(speed)
            acc_list.append(acc)
    plt.plot(speed_list, acc_list, 'b.')
plt.show()
