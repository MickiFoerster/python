import numpy as np


def f(x: np.float16):
    return np.sin(x)

def d_f(x: np.float16):
    return np.cos(x)

