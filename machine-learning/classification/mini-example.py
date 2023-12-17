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

def step(alpha, theta, A, y):
    sum = 0.
    for i in range(0, len(y)):
        deriv = (y[i] - h(theta, A[i]))
        sum += (y[i] - h(theta, A[i]))
    a = theta[0] + alpha * sum
    sum = 0.
    for i in range(0, len(y)):
        deriv = (y[i] - h(theta, A[i])) * A[i][1]
        sum += (y[i] - h(theta, A[i])) * A[i][1]
    b = theta[1] + alpha * sum

    return np.array([a, b])


def likelihood(y, theta, A):
    prod = 1.
    for i in range(0, len(y)):
        prod *= h(theta, A[i])**(y[i]) * (1-h(theta, A[i]))**(1-y[i])

    return prod

def log_likelihood(y, theta, A):
    sum = 0.
    for i in range(0, len(y)):
        x = A[i]
        sum += y[i] * math.log( h(theta, x) ) + (1-y[i]) * math.log( 1-h(theta, x) )

    return sum


alpha = 0.1
theta = np.array([-1, -1])
print(theta)
print("log l(theta): {}".format(log_likelihood(y, theta, A)))

for i in range(1, 1000):
    t = step(alpha, theta, A, y)
    theta = t
    print(theta)
    print("log l(theta): {}".format(log_likelihood(y, theta, A)))

print(h(theta, A[0]))
print(h(theta, A[1]))
print(h(theta, A[2]))
print(h(theta, A[3]))
