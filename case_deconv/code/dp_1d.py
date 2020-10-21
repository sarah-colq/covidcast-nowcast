"""Implementation of Nick Johnson's DP solution for 1d fused lasso."""
import matplotlib.pyplot as plt
import numpy as np


# from ctypes import *


def dp_1d(y, lam):
    n = y.shape[0]
    beta = np.zeros(n)

    # knots
    x = np.zeros((2 * n))
    a = np.zeros((2 * n))
    b = np.zeros((2 * n))

    # knots of back-pointers
    tm = np.zeros((n - 1))
    tp = np.zeros((n - 1))

    # step through first iteration manually
    tm[0] = -lam + y[0]
    tp[0] = lam + y[0]
    l = n - 1
    r = n
    x[l] = tm[0]
    x[r] = tp[0]
    a[l] = 1
    b[l] = -y[0] + lam
    a[r] = -1
    b[r] = y[0] + lam
    afirst = 1
    bfirst = -y[1] - lam
    alast = -1
    blast = y[1] - lam
    # now iterations 2 through n-1
    for k in range(1, n - 1):
        # compute lo
        alo = afirst
        blo = bfirst
        lo = l
        while lo <= r:
            if alo * x[lo] + blo > -lam:
                break
            alo += a[lo]
            blo += b[lo]
            lo += 1

        # compute hi
        ahi = alast
        bhi = blast
        hi = r
        while hi >= lo:
            if (-ahi * x[hi] - bhi) < lam:
                break
            ahi += a[hi]
            bhi += b[hi]
            hi -= 1

        # compute the negative knot
        tm[k] = (-lam - blo) / alo
        l = lo - 1
        x[l] = tm[k]

        # compute the positive knot
        tp[k] = (lam + bhi) / (-ahi)
        r = hi + 1
        x[r] = tp[k]

        # update a and b
        a[l] = alo
        b[l] = blo + lam
        a[r] = ahi
        b[r] = bhi + lam
        afirst = 1
        bfirst = -y[k + 1] - lam
        alast = -1
        blast = y[k + 1] - lam

    # compute the last coefficient - function has zero derivative here
    alo = afirst
    blo = bfirst
    for lo in range(l, r + 1):
        if alo * x[lo] + blo > 0:
            break
        alo += a[lo]
        blo += b[lo]

    beta[n - 1] = -blo / alo

    # compute the rest of the coefficients
    for k in range(n - 2, -1, -1):
        if beta[k + 1] > tp[k]:
            beta[k] = tp[k]
        elif beta[k + 1] < tm[k]:
            beta[k] = tm[k]
        else:
            beta[k] = beta[k + 1]
    return beta


if __name__ == "__main__":
    np.random.seed(12321)
    n = 100
    x = np.linspace(-2 * np.pi, 2 * np.pi, n)
    y = 1.5 * np.sin(x) + np.sin(2 * x) + np.random.randn(n) * np.sqrt(0.2)
    lam = 0.5

    beta = dp_1d(y, lam)
    # fun = CDLL("libfun.so")
    # fun.read.argtypes = c_int, POINTER(c_double), c_double, POINTER(c_double)
    # fun.read.restype = None
    # beta2 = (c_double * n)()
    # y_c = (c_double * n)(*y)
    # fun.tf_dp(c_int(n), y_c, c_double(lam), beta2)
    plt.scatter(x, y)
    plt.plot(x, beta, c="green")
    plt.show()
    plt.clf()
