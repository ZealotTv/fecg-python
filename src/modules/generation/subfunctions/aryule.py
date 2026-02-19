from numpy import concatenate, correlate
from numpy.linalg import solve
from scipy.linalg import toeplitz


def aryule(x, order):
    r = correlate(x, x, mode="full")
    r = r[len(r) // 2 :]
    R = toeplitz(r[:order])
    rhs = r[1 : order + 1]
    a = solve(R, rhs)
    a = concatenate(([1], -a))
    return a
