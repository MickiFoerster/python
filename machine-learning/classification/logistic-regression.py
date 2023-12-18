import math
import sys
import numpy as np
import pandas as pd
import logging
import signal

alpha = 13.8
theta = np.array([-30.85873244, 61.87780892])

def signal_handler(sig, frame):
    print("signal handler caught ctrl+c")
    print(f"last value of theta: {theta}")
    print("training data:")
    print(h(theta, A[0]))
    print(h(theta, A[1]))
    print(h(theta, A[2]))
    print(h(theta, A[3]))

    print("new data: 0.2")
    print(h(theta, np.array([1., 0.2])))
    print("new data: 0.8")
    print(h(theta, np.array([1., 0.8])))

    sys.exit(0)

def sigmoid(x):
    d = 1 + math.exp(-x)
    return 1 / d

def h(theta, x):
    theta_dot_x = np.dot(theta, x)

    return sigmoid(theta_dot_x)


class MaxLikelihoodFile:
    def __init__(self, filename):
        self.file = open(filename, "w")

    def __enter__(self):
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def __del__(self):
        # Destructor to ensure the file is closed if not explicitly closed
        if hasattr(self, 'file') and self.file:
            self.file.close()



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

def log_likelihood(y, theta, A):
    sum = 0.
    for i in range(0, len(y)):
        x = A[i]
        sum += y[i] * math.log( h(theta, x) ) + (1-y[i]) * math.log( 1-h(theta, x) )

    return sum


def read_input():
    csv_data = pd.read_csv("data.csv", dtype=np.float64)
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
        sum += (y[i] - h_value) * x[j]

    return sum

def batch_gradient_descent(alpha, A, y):
    global theta

    with MaxLikelihoodFile("max-likelihood_gradient_descent.log") as log_file:
        counter = 1

        l = log_likelihood(y, theta, A)
        log_file.write(f"{l}\n")
        old_likelihood = l

        while True:
            new_theta = theta.copy()
            for j in range(0, len(theta)):
                deriv = derivative_with_respect_to_theta_j(theta, A, y, j)
                new_theta[j] = theta[j] + alpha * deriv

            # stop when theta converges
            if np.array_equal(theta, new_theta):
                log.info("no difference in new theta value, so stop gradient descent")
                break

            theta = new_theta.copy()

            old_likelihood = l
            l = log_likelihood(y, theta, A)
            if old_likelihood >= l:
                log.info("likelihood is not increasing, so stop minimizing process")
                break

            log.info(f"new theta: {theta}, likelihood after {counter} iterations: {l}")
            log_file.write(f"{l}\n")

            counter += 1

        print(f"Batch Gradient descent converged after {counter} iterations with value: {theta}")

        return theta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    signal.signal(signal.SIGINT, signal_handler)

    log.info('reading input')
    A, y = read_input()
    for i in range(0, len(A)):
        for j in range(0, len(A[i])):
            assert type(A[i][j]) == np.float64
    for i in range(0, len(y)):
        assert type(y[i]) == np.float64
    for i in range(0, len(theta)):
        assert type(theta[i]) == np.float64

    batch_gradient_descent(alpha, A, y)
    print("Batch gradient descent solution: theta = {}" .format(tmp_theta))


