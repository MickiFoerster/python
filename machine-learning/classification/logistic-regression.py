import math
import sys
import numpy as np
import pandas as pd
import logging

def sigmoid(x):
    d = 1 + math.exp(-x)
    return 1 / d

def h(theta, x):
    theta_dot_x = np.dot(theta, x)

    return sigmoid(theta_dot_x)


# Likelihood: 
#
# L(theta) = P( y | x; theta ) 
#
# = Prod_(i=0)^(n) P( y^(i) | x^(i); theta )  (due to iid)
#
# = Prod_(i=0)^(n) h(x^(i))^(y^(i)) * ( 1 - h(x^(i))^(1-y^(i)) )
#
# log L(theta) = Sum_(i=0)^(n) y^(i) * log(h(x^(i))) + (1-y^(i)) * log(1-h(x^(i))

def likelihood(y, theta, A):
    prod = 1.
    for i in range(0, len(y)):
        prod *= h(theta, A[i])**(y[i]) * (1-h(theta, A[i]))**(1-y[i])

    return prod

def log_likelihood(y, theta, x):
    m = len(y)
    sum = 0.
    for i in range(0..m):
        sum += y[i]*math.log(h(x)) + (1-y[i]) * math.log(1-h(x))

    return prod


def read_input():
    csv_data = pd.read_csv("data.csv")
    x = csv_data.iloc[:,  0].values
    y = csv_data.iloc[:, -1].values

    # First column of feature matrice consists of ones
    first_column = np.ones(len(x))

    # feature matrix
    A = np.stack((first_column, x), axis=1)

    return (A, y)

def derivative_with_respect_to_theta_j(theta, A, y, j):
    sum = 0
    for i in range(0, len(y)):
        x = A[i]
        h_value = h(theta, x)
        #log.debug(f"h(x): {h_value}")
        sum += (y[i] - h_value) * x[j]

    log.debug(sum)

    return sum


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger(__name__)

    log.info('reading input')
    A, y = read_input()

    theta = np.array([0, 0])
    alpha = 0.01

    l = likelihood(y, theta, A)
    print(l)
    print(theta)

    new_theta = np.array([0, 0])
    for j in [0, 1]:
        sum = derivative_with_respect_to_theta_j(theta, A, y, 0)
        new_theta[j] = theta[j] + alpha * sum

    theta = new_theta
    l = likelihood(y, theta, A)
    print(l)
    print(theta)

    for j in [0, 1]:
        sum = derivative_with_respect_to_theta_j(theta, A, y, 0)
        new_theta[j] = theta[j] + alpha * sum

    theta = new_theta
    l = likelihood(y, theta, A)
    print(l)
    print(theta)

    for j in [0, 1]:
        sum = derivative_with_respect_to_theta_j(theta, A, y, 0)
        new_theta[j] = theta[j] + alpha * sum

    theta = new_theta
    l = likelihood(y, theta, A)
    print(l)
    print(theta)


