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
f = open(sys.argv[1], 'r')

cmd_table = {}

for line in f:
    items = line.split(',')
    cmd = round(float(items[0]))
    speed = float(items[1])
    acc = float(items[2])
    if cmd in cmd_table:
        speed_table = cmd_table[cmd]
        if speed in speed_table:
            speed_table[speed].append(acc)
        else:
            speed_table[speed] = [acc]
    else:
        speed_table = {}
        speed_table[speed] = [acc]
        cmd_table[cmd] = speed_table
f.close()

NCURVES = len(cmd_table)
np.random.seed(101)
curves = [np.random.random(20) for i in range(NCURVES)]
values = range(NCURVES)
jet = cm = plt.get_cmap('brg')
cNorm = mcolors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

cnt = 0
cmds = cmd_table.keys()
cmds.sort()

fig, ax = plt.subplots()
for cmd in cmds:
    print "ctrl cmd = ", cmd
    speed_table = cmd_table[cmd]
    X = []
    Y = []
    speeds = speed_table.keys()
    speeds.sort()
    for speed in speeds:
        X.append(speed)
        Y.append(np.mean(speed_table[speed]))
    colorVal = scalarMap.to_rgba(values[cnt])
    ax.plot(
        X,
        Y,
        c=colorVal,
        linestyle=':',
        marker=markers[cnt % len(markers)],
        label="cmd=" + str(cmd))
    cnt += 1

ax.legend(loc='upper center', shadow=True, bbox_to_anchor=(0.5, 1.1), ncol=5)

plt.ylabel("acc")
plt.xlabel("speed")
plt.grid()
plt.show()
