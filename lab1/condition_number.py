import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from lab1.gradient_descent import gradient_descent

def create_matrix(k, n):
    r = sqrt(k)
    A = np.random.randn(n, n)
    u, s, v = np.linalg.svd(A)
    h, l = np.max(s), np.min(s)
    new_s = h * (1 - ((r - 1) / r) / (h - l) * (h - s))
    new_A = (u * new_s) @ v.T
    new_A = new_A @ new_A.T
    assert np.isclose(np.linalg.cond(new_A), k)
    return new_A

def number_of_iters(cond, n_vars, step_chooser, n_checks=100):
    avg_iters = 0
    for _ in range(n_checks):
        A = create_matrix(cond, n_vars)
        b = np.random.randn(len(A))
        init_x = np.random.randn(len(A))
        f = lambda x: x.dot(A).dot(x) - b.dot(x)
        f_grad = lambda x: (A + A.T).dot(x) - b
        trace = gradient_descent(f, f_grad, init_x, step_chooser, 'value')
        avg_iters += len(trace)
    return avg_iters / n_checks

n_vars = [5, 10, 20, 50, 100]
condition_numbers = np.linspace(1, 1000, 50)
for var in n_vars:
    iters = list(map(lambda cond: number_of_iters(cond, var), condition_numbers))
    plt.plot(condition_numbers, iters, label='n=' + str(var))

plt.xlabel('Число обусловленности')
plt.ylabel('Число итераций')
plt.legend()
plt.savefig('iterations.pdf')
plt.show()
