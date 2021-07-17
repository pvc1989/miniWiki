import numpy as np
from numpy.ma.core import innerproduct
from scipy.optimize import root

class Hexa(object):

    _x_local_i = np.array([-1, +1, +1, -1, -1, +1, +1, -1])
    _y_local_i = np.array([-1, -1, +1, +1, -1, -1, +1, +1])
    _z_local_i = np.array([-1, -1, -1, -1, +1, +1, +1, +1])
    _xyz_local_3x8 = np.array([_x_local_i, _y_local_i, _z_local_i])

    _gauss_points = np.array([
        -np.sqrt((3 - 2 * np.sqrt(6/5) / 7)),
        +np.sqrt((3 - 2 * np.sqrt(6/5) / 7)),
        -np.sqrt((3 + 2 * np.sqrt(6/5) / 7)),
        +np.sqrt((3 + 2 * np.sqrt(6/5) / 7)),
    ])

    _gauss_weights = np.array([
        (18 + np.sqrt(30)) / 36,
        (18 + np.sqrt(30)) / 36,
        (18 - np.sqrt(30)) / 36,
        (18 - np.sqrt(30)) / 36,
    ])

    def __init__(self, x, y, z):
        self._xyz_global_3x8 = np.array([x, y, z])
        assert(self._xyz_global_3x8.shape == (3,8))

    @classmethod
    def _shape_8x1(cls, x_local, y_local, z_local):
        n_vec = (1 + cls._x_local_i * x_local) * (1 + cls._y_local_i * y_local) * (1 + cls._z_local_i * z_local) / 8
        return n_vec.reshape(8, 1)

    @classmethod
    def _shape_local_der_8x3(cls, x_local, y_local, z_local): 
        return np.array([
            cls._x_local_i * (1 + cls._y_local_i * y_local) * (1 + cls._z_local_i * z_local),
            cls._y_local_i * (1 + cls._x_local_i * x_local) * (1 + cls._z_local_i * z_local),
            cls._z_local_i * (1 + cls._x_local_i * x_local) * (1 + cls._y_local_i * y_local)
        ]).T / 8

    def local_to_global(self, x_local, y_local, z_local):
        return self._xyz_global_3x8.dot(self._shape_8x1(x_local, y_local, z_local)).reshape(3)

    def global_to_local(self, x_global, y_global, z_global):
        rhs = np.array([x_global, y_global, z_global])
        func = lambda xyz_local: self.local_to_global(xyz_local[0], xyz_local[1], xyz_local[2]) - rhs
        sol = root(func, np.zeros(3), jac=lambda xyz_local: self.jacobian(xyz_local[0], xyz_local[1], xyz_local[2]))
        return sol.x

    def jacobian(self,  x_local, y_local, z_local):
        mat_j = self._xyz_global_3x8.dot(self._shape_local_der_8x3(x_local, y_local, z_local))
        return mat_j

    def gauss_quadrature(self, f_local):
        sum = f_local(0, 0, 0) * 0
        for i in range(4):
            weight = self._gauss_weights[i]
            x_local = self._gauss_points[i]
            for j in range(4):
                weight *= self._gauss_weights[j]
                y_local = self._gauss_points[j]
                for k in range(4):
                    weight *= self._gauss_weights[k]
                    z_local = self._gauss_points[k]
                    sum += weight * f_local(x_local, y_local, z_local)
        return sum

    def integrate(self, integrand_global):
        def integrand_local(x_local, y_local, z_local):
            x_global, y_global, z_global = self.local_to_global(x_local, y_local, z_local)
            f_val = integrand_global(x_global, y_global, z_global)
            det_j = np.linalg.det(self.jacobian(x_local, y_local, z_local))
            return f_val * det_j
        return self.gauss_quadrature(lambda z, y, x: integrand_local(x, y, z))

    def innerprod(self, f, g):
        return self.integrate(lambda x, y, z: f(x, y, z) * g(x, y, z))

    def norm(self, f):
        return np.sqrt(self.innerprod(f, f))

    def orthonormalize(self, raw_basis):
        n = len(raw_basis(0, 0, 0))
        def integrand_global(x, y, z):
            column = raw_basis(x, y, z)
            return column.dot(column.T)
        A = self.integrate(integrand_global)    
        assert(A.shape == (n, n))
        S = np.eye(n)
        S[0, 0] = 1 / np.sqrt(A[0, 0])
        for i in range(1, n):
            for j in range(i):
                temp = 0
                for k in range(j+1):
                    temp += S[j, k] * A[k, i]
                for l in range(j+1):
                    S[i, l] -= temp * S[j, l]
            norm_sq = 0
            for j in range(i+1):
                for k in range(j):
                    norm_sq += 2 * S[i, j] * S[i, k] * A[k, j]
            for j in range(i+1):
                norm_sq += S[i, j] * S[i, j] * A[j, j]
            S[i] /= np.sqrt(norm_sq)
        return S

if __name__ == '__main__':
    x = np.array([-1, +1, +1, -1, -1, +1, +1, -1])
    y = np.array([-1, -1, +1, +1, -1, -1, +1, +1])
    z = np.array([-1, -1, -1, -1, +1, +1, +1, +1])
    hexa = Hexa(x+1, y+1, z+1)

    print(hexa.global_to_local(1, 1, 1))
    print(hexa.global_to_local(1.5, 1.5, 1.5))
    print(hexa.global_to_local(3, 4, 5))
    
    print(hexa.integrate(lambda x, y, z: 3))

    def raw_basis(x, y, z):
        return np.array([
            1,
            x, y, z,
            x * x, x * y, x * z,
            y * y, y * z, z * z,
        ]).reshape(10,1)
    
    schmidt = hexa.orthonormalize(raw_basis)
    print('schmidt = ')
    print(schmidt)
    orthonormal_basis = lambda x, y, z: schmidt.dot(raw_basis(x, y, z))
    def mat_for_test(x, y, z):
        column = orthonormal_basis(x, y, z)
        return column.dot(column.T)
    print(hexa.integrate(mat_for_test))
