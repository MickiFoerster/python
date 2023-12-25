#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure()

x = np.linspace(0, 1, 2)
x = np.arange(0, 1, 0.1)
print(x)

def f(x):
    theta = np.array([-141.70329858, 415.15121384, -297.95582757])
    return theta[0]+theta[1]*x+theta[2]*x**2

y = np.zeros(len(x))
for i in range(0, len(x)):
    y[i] = f(x[i])

axes = fig.add_axes([0.1, 0.2, 0.6, 0.8])
axes.plot(x, y, 'b')

plt.xlabel('x values')
plt.ylabel('y values')

import pandas as pd
try:
    dataset = pd.read_csv("data.csv")
except:
    print('error: could not read CSV file')
    sys.exit(1)

train_x = dataset.iloc[:, :-1].values
train_y = dataset.iloc[:, -1].values

plt.scatter(train_x, train_y, color = 'red')

plt.show()
