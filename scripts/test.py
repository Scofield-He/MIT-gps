#! python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


file = 'C:\\tmp\\SolarWind.csv'
df = pd.read_csv(file).query('year == 2015 & Hour == 5')
print(len(df))

f = np.linspace(0.1, 30, 3000)
doy = np.array([float(_) for _ in df['doy']])
Vs = np.array([float(_) for _ in df['Solar_Wind_Speed']])
normval = doy.shape[0]
pgram = signal.lombscargle(doy, Vs, f)

figure = plt.figure(figsize=(12, 4))
ax1 = figure.add_subplot(211)
ax1.plot(doy, Vs, 'bx', ms=4, alpha=0.8)

ax2 = figure.add_subplot(212)
ax2.plot(f, np.sqrt(4 * (pgram / normval)))
ax2.xaxis.set_major_locator(MultipleLocator(5))
ax2.xaxis.set_minor_locator(MultipleLocator(1))
plt.show()
