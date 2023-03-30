import numpy as np 
from numpy import pi, sin, log
from scipy.stats import linregress
from scheme_from_taylor import Scheme


class AccuracyMeasurer(object):

    def __init__(self, scheme):
        self._scheme = scheme
        self._accuracy = np.zeros(scheme.n_orders())
        self._accuracy[0] = np.inf
        steps = 2*pi / 2**np.arange(start=4, stop=10)
        errors = np.zeros((scheme.n_orders() , len(steps)))
        for i in range(len(steps)):
            h = steps[i]
            scheme.reset_step(h)
            x = np.arange(start=0, stop=2*pi, step=h)
            u = self._exact_pth_derivative(x, p=0)
            for p in range(1, scheme.n_orders()):
                exact_pth_dv = self._exact_pth_derivative(x, p)
                approx_pth_dv = self._approx_pth_derivative(u, p, scheme)
                errors[p][i] = abs(exact_pth_dv - approx_pth_dv).mean()
        for p in range(1, scheme.n_orders()):
            self._accuracy[p] = linregress(log(steps), log(errors[p])).slope

    @staticmethod
    def _exact_pth_derivative(x, p):
        # f(x) = \sin(kx), f^{(p)}(x) = k^p \sin(kx + p\pi/2)
        return sin(x + p*pi/2)

    @staticmethod
    def _approx_pth_derivative(u, p, scheme):
        n = len(u)
        v = np.zeros(n)
        for i in range(n):
            index = (scheme.i_shift() + i) % n
            v[i] = scheme.coefficient(p).dot(u[index])
        return v

    def accuracy(self, p):
        return self._accuracy[p]


if __name__ == '__main__':
    scheme = Scheme(i_shift=(-5, -4, -3, -2, -1, 0, 1))
    measurer = AccuracyMeasurer(scheme)
    for order in range(scheme.n_orders()):
        print(order, "%4.1f"%measurer.accuracy(order))