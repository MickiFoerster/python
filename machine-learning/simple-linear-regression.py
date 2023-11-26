#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import logging

alpha = 0.0001
#theta = np.array([25813, 9446])
#theta = np.array([10000, 10000])
theta = np.array([0, 0])

# three solution methods
# 1) batch gradient descent
# 2) stochastic gradient descent
# 3) Newton Raphson


def h(theta, x):
    return np.dot(theta, x)

def read_input():
    csv_data = pd.read_csv("Salary_Data.csv")
    x = csv_data.iloc[:,  0].values
    y = csv_data.iloc[:, -1].values

    # First column of feature matrice consists of ones
    first_column = np.ones(len(x))

    # feature matrix
    A = np.stack((first_column, x), axis=1)

    return (A, y)

def cost_function(theta, A, y):
    sum = 0
    for i in range(0, len(y)):
        x = A[i]
        h_value = h(theta, x)
        sum = sum + ( h_value - y[i] )**2
    sum /= 2
    return sum

# ( h(x) - y ) * x_j
def get_sum(theta, A, y, j):
    sum = 0
    for i in range(0, len(y)):
        x = A[i]
        h_value = h(theta, A[i])
        log.debug(f"h(x): {h_value}")
        sum = sum + ( h_value - y[i] ) * A[i][j]
        log.debug(f"sum: {sum}")

    return sum


def batch_gradient_descent(theta, A, y):
    counter = 1
    while True:
        new_theta = np.array([0,0])
        for j in [0, 1]:
            sum = get_sum(theta, A, y, j)
            new_theta[j] = theta[j] - alpha * sum

        # stop when theta converges
        if np.array_equal(theta, new_theta):
            log.info("no difference to last iteration, so stop gradient descent")
            break

        theta = new_theta
        log.debug("new theta: {:.2f}, {:.2f}".format(theta[0], theta[1]))
        counter += 1

    print(f"Batch Gradient descent converged after {counter} iterations with value: {theta}")

    return theta

def stochastic_gradient_descent(theta, A, y):
    global cost_log_file
    new_theta = np.array([0,0])

    counter = 1
    cost = cost_function(theta, A, y)

    while True:
        new_theta= np.array([0, 0])
        for i in range(0, len(y)):
            for j in [0, 1]:
                new_theta[j] = theta[j] - alpha * (h(theta, A[i]) - y[i]) * A[i][j]

        if np.array_equal(theta, new_theta):
            log.info("no difference to last iteration, so stop gradient descent")
            break

        theta = new_theta
        log.debug("new theta: {:.2f}, {:.2f}".format(theta[0], theta[1]))

        old_cost = cost
        cost = cost_function(theta, A, y)
        if old_cost < cost:
            log.info("cost function is not decreasing, so stop minimizing process")
            break
        log.debug(f"cost function value: {cost}")
        cost_log_file.write(f"{cost}\n")

        counter += 1

    print(f"Stochastic Gradient descent converged after {counter} iterations with value: {theta}")

    return theta

def newton_raphson(theta, A, y):
    global cost_log_file

    counter = 1
    cost = cost_function(theta, A, y)
    cost_log_file.write(f"{cost}\n")

    while True:
        # compute gradient of J(theta): gradJ 
        gradJ = np.array([
            get_sum(theta, A, y, 0),
            get_sum(theta, A, y, 1),
        ])

        # compute Hessian of J(theta):  H
        m = len(y)
        # sum from i=1 to m over x_1^(i)
        sum1 = 0
        for i in range(0, len(y)):
            sum1 += A[i][1]

        # sum from i=1 to m over x_1^((i)*2)
        sum2 = 0
        for i in range(0, len(y)):
            sum2 += A[i][1]**2

        H = np.array([
            [m,    sum1],
            [sum1, sum2], 
        ])

        # theta^(t+1) = theta^(t) - H^-1 * gradJ
        # <=>  H * theta^(t+1) = H * theta^(t) - gradJ
        #          ^--- unknown => linear equation system to solve
        # b = H * theta^(t) - gradJ
        # solve linear equation system H x = b

        log.debug(f"grad J: {gradJ}")
        log.debug(f"H: {H}")
        log.debug(f"theta: {theta}")
        log.debug(f"theta: {gradJ}")
        b = np.subtract(H.dot(theta), gradJ)
        log.debug(f"rhs of lin. eq. system: {b}")

        log.debug("solve linear equuation system ...")
        x = np.linalg.solve(H, b)
        log.debug(f"solution: {x}")

        old_cost = cost
        cost = cost_function(x, A, y)
        log.debug(f"cost function value: {cost}")
        cost_log_file.write(f"{cost}\n")
        if old_cost <= cost:
            log.info(f"cost function is not decreasing ({old_cost} < {cost}), so stop minimizing process")
            break

        # new theta is x
        if np.array_equal(theta, x):
            log.info("no difference to last iteration, so stop newton raphson")
            #break
            if counter == 4:
                break

        theta = x
        log.debug("new theta: {:.2f}, {:.2f}".format(theta[0], theta[1]))

        counter += 1
    log.info(f"Newton Raphson stopped after {counter} iterations")

    return theta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    cost_log_file = open('cost-function.log', 'w')

    log.info('reading input')
    A, y = read_input()
    log.info('input data has been read')

    tmp_theta = theta.copy()
    tmp_theta = batch_gradient_descent(theta, A, y)
    print("Batch gradient descent solution: theta = [{}, {}]"
          .format(tmp_theta[0], tmp_theta[1]))
    print("Cost function value: {}"
          .format(cost_function(tmp_theta, A, y)))

    tmp_theta = theta.copy()
    tmp_theta = stochastic_gradient_descent(tmp_theta, A, y)
    print("Stochastic gradient descent solution: tmp_theta = [{}, {}]".format(tmp_theta[0], tmp_theta[1]))
    print("Cost function value: {}"
          .format(cost_function(tmp_theta, A, y)))

    tmp_theta = theta.copy()
    tmp_theta = newton_raphson(tmp_theta, A, y)
    print("Newton method solution: tmp_theta = [{}, {}]".format(tmp_theta[0], tmp_theta[1]))
    print("Cost function value: {}"
          .format(cost_function(tmp_theta, A, y)))

    cost_log_file.close()
