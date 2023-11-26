import numpy as np

A = np.array([[2, 1],
              [1, -3]])
b = np.array([8, -3])

x = np.linalg.solve(A, b)

print("Solving Ax=b result:")
print("x = [{} {}]".format(x[0], x[1]))

