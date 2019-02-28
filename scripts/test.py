#! python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dateparse(dates):
    return pd.datetime.strptime(dates, '%Y-%m-%d')


x = np.array(list(range(30)))
y = np.sin(x) + np.random.random(30) - 0.5

figure = plt.figure()
ax1 = figure.add_subplot(211)
ax1.plot(x, y)

ax2 = figure.add_subplot(212, sharex=ax1)
# ax2 = ax1.twinx()
x2 = np.linspace(0, 30, 61)
print(x2)
y2 = np.cos(x2)
ax2.plot(x2, y2)
plt.show()
