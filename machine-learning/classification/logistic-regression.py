import math
import sys
import numpy as np
import pandas as pd
import logging
import signal

alpha = 1.1
#theta = np.array([-466.60366825, 1355.00942471, -965.20220283])
theta = np.array([1., 1.])

def signal_handler(sig, frame):
    print("signal handler caught ctrl+c")
    print_summary_at_the_end()

def print_summary_at_the_end():
    print(f"last value of theta: {theta}")
    print("training data:")
    print(h(theta, A[0]))
    print(h(theta, A[1]))
    print(h(theta, A[2]))
    print(h(theta, A[3]))

    print("new data: 0.45")
    print(h(theta, np.array([1., 0.45])))
    print("new data: 0.55")
    print(h(theta, np.array([1., 0.55])))
    print("new data: 0.65")
    print(h(theta, np.array([1., 0.65])))
    print("new data: 0.75")
    print(h(theta, np.array([1., 0.75])))
    print("new data: 0.85")
    print(h(theta, np.array([1., 0.85])))

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
        h_theta_of_x = h(theta, x)
        try:
            sum += y[i] * math.log( h_theta_of_x ) + (1-y[i]) * math.log( 1-h_theta_of_x )
        except:
            print_summary_at_the_end()

    return sum


def read_input():
    csv_data = pd.read_csv("data.csv", dtype=np.float64)
    x = csv_data.iloc[:,  0].values
    y = csv_data.iloc[:, -1].values

    # First column of feature matrice consists of ones
    first_column = np.ones(len(x))
   
    #squared = np.zeros(len(x))
    #for i in range(0, len(y)):
    #    squared[i] = x[i]**2

    #cubic = np.zeros(len(x))
    #for i in range(0, len(y)):
    #    cubic[i] = x[i]**3
    #
    #grade_four = np.zeros(len(x))
    #for i in range(0, len(y)):
    #    grade_four[i] = x[i]**3
    
    # feature matrix
    #A = np.stack((first_column, x), axis=1)
    #A = np.stack((first_column, x, squared, cubic, grade_four), axis=1)
    A = np.stack((first_column, x), axis=1)

    return (A, y)

def derivative_with_respect_to_theta_j(theta, A, y, j):
    sum = 0
    for i in range(0, len(y)):
        x = A[i]
        derivative_with_respect_to_theta_j = 0
        if j == 0:
            derivative_with_respect_to_theta_j = x[0]
        elif j == 1:
            derivative_with_respect_to_theta_j = x[1]
        #elif j == 2:
        #    derivative_with_respect_to_theta_j = x[1]**2
        #elif j == 3:
        #    derivative_with_respect_to_theta_j = x[1]**3
        #elif j == 4:
        #    derivative_with_respect_to_theta_j = x[1]**4

        h_value = h(theta, x)
        sum += (y[i] - h_value) * derivative_with_respect_to_theta_j

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

            if counter == 10000:
                print("stop after 10.000 iterations")
                break

            counter += 1

        print(f"Batch Gradient descent converged after {counter} iterations with value: {theta}")

        return theta

def GradientMaxLikelihood(theta, A, y):
    gradient = np.zeros(len(theta))

    for j in range(0, 2):
        sum = 0.
        for i in range(0, len(y)):
            x = A[i]
            h_of_x_i = h(theta, x)
            sum += (y[i] - h_of_x_i) * x[j]

        gradient[j] = sum

    return gradient

def Hessian(theta, A):
    sum = 0.
    for i in range(0, len(y)):
        x = A[i]
        h_of_x_i = h(theta, x)
        sum += h_of_x_i * (1 - h_of_x_i) 
    h00 = -sum

    sum = 0.
    for i in range(0, len(y)):
        x = A[i]
        h_of_x_i = h(theta, x)
        sum += h_of_x_i * (1 - h_of_x_i) * x[1]
    h01 = -sum
    h10 = h01

    sum = 0.
    for i in range(0, len(y)):
        x = A[i]
        h_of_x_i = h(theta, x)
        sum += h_of_x_i * (1 - h_of_x_i) * x[1]**2
    h11 = -sum

    return np.array([
        [h00, h01], 
        [h10, h11],
    ])

def NewtonRaphson(A, y):
    global theta

    with MaxLikelihoodFile("max-likelihood-newton.log") as log_file:
        l = log_likelihood(y, theta, A)
        log_file.write(f"{l}\n")

        while True:
            gradient = GradientMaxLikelihood(theta, A, y)
            hessian = Hessian(theta, A)

            b = np.subtract(hessian.dot(theta), gradient)
            # solve linear system of equations
            x = np.linalg.solve(hessian, b)

            new_theta = x
            old_l = l
            l = log_likelihood(y, new_theta, A)
            log_file.write(f"{l}\n")

            if old_l >= l:
                log.info(f"likelihood is not increasing anymore, so stop iterating")
                break

            log.info(f"new theta: {new_theta}")
            theta = new_theta

        return new_theta


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

    tmp_theta = batch_gradient_descent(alpha, A, y)
    print("Batch gradient descent solution: theta = {}" .format(tmp_theta))

    print("Now start Newton Raphson method:")
    theta = np.array([1., 1.])
    new_theta = NewtonRaphson(A, y)
    log.info(f"theta: {theta}")

