import numpy
from scipy.linalg import cho_factor, cho_solve

def deltaTest(eps, xkv, xk1v, fxkv, fxk1v, dkv):
    return np.linalg.norm(dkv) < eps

def argTest(eps, xkv, xk1v, fxkv, fxk1v, dkv):
    return np.linalg.norm(xk1v - xkv) < eps

def valueTest(eps, xkv, xk1v, fxkv, fxk1v, dkv):
    return abs(fxk1v - fxkv) < eps

def newtonMethod(f, startX, grad, hess, stopTest, maxIters=100):
    xkv = startX
    fxkv = f(xkv)
    trace = [xkv]
    while True:
        gradv = grad(xkv)
        hessv = hess(xkv)
        hessiv = cho_solve(cho_factor(hessv), np.eye(hessv.shape[0]))
        akv = 1
        dkv = np.matmul(gradv, hessiv)
        xk1v = xkv - akv * dkv
        fxk1v = f(xk1v)
        trace.append(xk1v)

        if len(trace) == maxIters:
            raise StopIteration()

        if stopTest(xkv, xk1v, fxkv, fxk1v, dkv):
            return trace
        xkv = xk1v
        fxkv = fxk1v
