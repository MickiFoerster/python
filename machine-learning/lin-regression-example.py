import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time import sleep
from threading import Thread

def h(theta, x):
    return theta[0]*x[0] + theta[1]*x[1]

def animate(i):
    global theta
    global trainings_set
    global y

    x = np.linspace(0, 10, 100)
    def f(theta, x):
        return theta[0] + theta[1]*x

    plt.scatter(trainings_set[:,1], y)
    plt.plot(x, f(theta, x))

def task():
    global theta
    global trainings_set
    global y

    for i in range(100):
        delta = np.array([
            h(theta, trainings_set[0]) - y[0],
            h(theta, trainings_set[1]) - y[1],
            h(theta, trainings_set[2]) - y[2],
            ])
        d_J_theta = np.array([
            (1/3) * (delta[0] + delta[1] + delta[2]),
            (1/3) * (delta[0]*trainings_set[0][1]
                     + delta[1]*trainings_set[1][1] 
                     + delta[2]*trainings_set[2][1]
            )])

        theta = np.array([
            theta[0] - alpha * d_J_theta[0], 
            theta[1] - alpha * d_J_theta[1]
            ])
        print("theta:", theta)
        sleep(0.5)

alpha = 0.3
trainings_set = np.array([
    [1,1],
    [1,2],
    [1,3],
    ])

y = np.array([1,2,3])

theta = np.array([0,0])

thread = Thread(target=task)
print("Now start the gradient descent task")
thread.start()

print("Plot the resulting linear line")
ani = animation.FuncAnimation(plt.figure(), animate, interval=500)
plt.show()

thread.join()
