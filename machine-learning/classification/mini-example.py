import math
import sys
import numpy as np

def g(z):
    d = 1 + math.exp(-z)
    return 1/d

def h(theta, x):
    theta_dot_x = np.dot(theta, x)
    return g(theta_dot_x)

y = np.array([0,0,1,1])
A = [
        [1, 0.1],
        [1, 0.3],
        [1, 0.7],
        [1, 0.9],
]


print(A)
print(y)

alpha = 0.1
theta = np.array([1, 1])
t = np.array([0, 0])

sum = 0.
for i in range(0, len(y)):
    print(A[i])
    sum += (y[i] - h(theta, A[i]))
t[0] = theta[0] + alpha * sum


sum = 0.
for i in range(0, len(y)):
    print(A[i][1])
    sum += (y[i] - h(theta, A[i])) * A[i][1]
t[1] = theta[1] + alpha * sum

print(theta)
print(t)
