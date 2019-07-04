import numpy as np 
from scipy.special import factorial
from scipy.linalg import solve


class Scheme(object):

    def __init__(self, i_shift, step=1):
        self._i_shift = np.array(i_shift)
        self._step = step
        n = len(i_shift)
        matrix = np.zeros((n,n))
        rhs = np.zeros((n,n))
        for p in range(n):
            matrix[p] = self._i_shift**p
            rhs[p][p] = factorial(p) / self._step**p
        self._coefficients = solve(matrix, rhs).transpose()

    def reset_step(self, step):
        for p in range(self.n_orders()):
            self._coefficients[p] *= (self._step / step)**p
        self._step = step

    def step(self):
        return self._step

    def i_shift(self):
        return self._i_shift
        
    def n_orders(self):
        return len(self._i_shift)

    def n_points(self):
        return self.n_orders()

    def coefficient(self, order_of_derivative):
        return self._coefficients[order_of_derivative]


if __name__ == '__main__':
    scheme = Scheme(i_shift=(-2, -1, 0), step=1)
    print("i_shift:", scheme.i_shift())

    print("h =", scheme.step())
    print("Order\t Coeffients")
    for order in range(scheme.n_orders()):
        print(order, '\t', scheme.coefficient(order))

    scheme.reset_step(step=0.5)
    print("h =", scheme.step())
    print("Order\t Coeffients")
    for order in range(scheme.n_orders()):
        print(order, '\t', scheme.coefficient(order))    