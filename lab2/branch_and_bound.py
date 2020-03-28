import numpy as np
import math
from scipy.optimize import linprog

def is_integer(v):
    return v.is_integer() or np.isclose(v, round(v))

def branch_and_bound_search(f, A, b, lub):
    num_vars = len(f)
    all_ints = True
    res = linprog(f, A, b, bounds=lub)
    X, v = res.x, res.fun
    for i in range(num_vars):
        if (not np.isnan(v)) and (not is_integer(X[i])):
            all_ints = False
            var_val = math.floor(X[i])
            var_ind = i

    if all_ints:
        if np.isnan(v):
            return (None, 1, inf)
        else:
            return (X, 1, v)
    else:
        lub1 = lub
        lub2 = lub
        lb, ub = lub[var_ind]
        lub1[var_ind] = (var_val + 1, ub)
        lub2[var_ind] = (lb, var_val)

        [X1, i1, v1] = branch_and_bound_search(f, A, b, lub1)
        [X2, i2, v2] = branch_and_bound_search(f, A, b, lub2)

        if v1 < v2:
            return (X1, i1 + i2, v1)
        else:
            return (X2, i1 + i2, v2)

def branch_and_bound(f, A, b):
    return branch_and_bound_search(f, A, b, [(0, np.inf)] * len(f))

f = [-13, -8]
A = [[1, 2], [5, 2]]
b = [10, 20]

[X, i, v] = branch_and_bound(f, A, b)

print("Values =", X)
print("Iteration =", i)
print("Function value =", v)