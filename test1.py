#! python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.signal import lombscargle
import matplotlib.pyplot as plt


A = 2.
w = 1.
phi = 0.5 * np.pi
nin, nout = 1000, 100000
frac_points = 0.9

r = np.random.rand(nin)
"""
count = 0
for _ in r:
    if _ > frac_points:
        count += 1
print(count)
"""
x = np.linspace(0.01, 10*np.pi, nin)
x = x[r >= frac_points]
normval = x.shape[0]
print(len(x))

y = A * np.sin(w*x + phi)

f = np.linspace(0.01, 10, nout)

pgram = lombscargle(x, y, f)

plt.subplot(311)
plt.plot(x, y, 'b+')

plt.subplot(312)
plt.plot(f, np.sqrt(4*(pgram/normval)))

plt.subplot(313)
plt.plot(f, pgram)
plt.show()
