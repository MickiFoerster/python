from functions import f, d_f
import numpy as np

def test_derivative_of_f():
    for v in np.array([0.0, 1.0, 5.0, 10.0], dtype=np.float16):
        assert np.cos(v) == d_f(v)
