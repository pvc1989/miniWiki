import abc
import unittest
import numpy as np
from scipy import sparse

class Element(abc.ABC):

    @abc.abstractstaticmethod
    def xyz(i):
        pass

    @abc.abstractstaticmethod
    def hierarchical_shape(i, x, y, z) -> float:
        pass

    @abc.abstractstaticmethod
    def hierarchical_shapes(n, x, y, z) -> np.ndarray:
        pass

    @abc.abstractstaticmethod
    def lagrange_shapes(n, x, y, z) -> np.ndarray:
        pass

    @staticmethod
    def p2n(p) -> int:
        return (p + 1) * (p + 2) * (p + 3) // 6

    @staticmethod
    def taylor_basis(p, x, y, z) -> np.ndarray:
        n = Element.p2n(p)
        values = np.ndarray(n)
        values[0] = 1
        if n > 1:
            values[1] = x
            values[2] = y
            values[3] = z
        if n == 4:
            return values
        for p_curr in range(2, p + 1):
            n_curr = Element.p2n(p_curr)
            n_prev = Element.p2n(p_curr - 1)
            n_prev_prev = Element.p2n(p_curr - 2)
            n_diff = n_prev - n_prev_prev
            for i in range(n_prev, n_prev + n_diff):
                values[i] = x * values[i - n_diff]
            for i in range(n_prev + n_diff, n_curr - 1):
                values[i] = y * values[i - n_diff - p_curr]
            values[n_curr - 1] = z * values[n_prev - 1]
        return values

    @staticmethod
    def taylor_to_lagrange(Lagrange, n_lagrange, degree):
        assert issubclass(Lagrange, Element)
        k = 11
        xyz = np.random.rand(k) * 2 - 1
        n_sample = k**3
        n_taylor = Element.p2n(degree)
        lhs = np.ndarray((n_sample, n_taylor))
        rhs = np.ndarray((n_sample, n_lagrange))
        i = 0
        for x in xyz:
            for y in xyz:
                for z in xyz:
                    lhs[i] = Element.taylor_basis(degree, x, y, z)
                    rhs[i] = Lagrange.lagrange_shapes(n_lagrange, x, y, z)
                    i += 1
        assert i == n_sample
        taylor_to_lagrange = np.linalg.solve(lhs.T @ lhs, lhs.T @ rhs)
        base = 2**(degree)
        for i in range(n_taylor):
            for j in range(n_lagrange):
                val = taylor_to_lagrange[i, j]
                if np.abs(val) > 1e-3:
                    taylor_to_lagrange[i, j] = np.round(val * base) / base
                else:
                    assert np.abs(val) < 1e-8, val
                    taylor_to_lagrange[i, j] = 0
        return taylor_to_lagrange


class Hexahedron(Element):

    _x = np.array([
      -1, 1, 1, -1,
      -1, 1, 1, -1,
      0, 1, 0, -1,
      0, 1, 0, -1,
      -1, 1, 1, -1,
      -1, 1, 0, 0, 0, 0, 0
    ])
    _y = np.array([
      -1, -1, 1, 1,
      -1, -1, 1, 1,
      -1, 0, 1, 0,
      -1, 0, 1, 0,
      -1, -1, 1, 1,
      0, 0, -1, 1, 0, 0, 0
    ])
    _z = np.array([
      -1, -1, -1, -1,
      1, 1, 1, 1,
      -1, -1, -1, -1,
      1, 1, 1, 1,
      0, 0, 0, 0,
      0, 0, 0, 0, -1, 1, 0
    ])

    @staticmethod
    def xyz(i: int):
        self = Hexahedron
        return self._x[i], self._y[i], self._z[i]

    @staticmethod
    def hierarchical_shape(i, x, y, z) -> float:
        self = Hexahedron
        if i < 8:
            return (1 + self._x[i] * x) * (1 + self._y[i] * y) * (1 + self._z[i] * z) / 8
        elif i in (8, 10, 12, 14):
            return (1 - x * x) * (1 + self._y[i] * y) * (1 + self._z[i] * z) / 4
        elif i in (9, 11, 13, 15):
            return (1 + self._x[i] * x) * (1 - y * y) * (1 + self._z[i] * z) / 4
        elif i in (16, 17, 18, 19):
            return (1 + self._x[i] * x) * (1 + self._y[i] * y) * (1 - z * z) / 4
        elif i < 22:
            return (1 + self._x[i] * x) * (1 - y * y) * (1 - z * z) / 2
        elif i < 24:
            return (1 - x * x) * (1 + self._y[i] * y) * (1 - z * z) / 2
        elif i < 26:
            return (1 - x * x) * (1 - y * y) * (1 + self._z[i] * z) / 2
        else:
            return (1 - x * x) * (1 - y * y) * (1 - z * z)

    @staticmethod
    def hierarchical_shapes(n, x, y, z) -> np.ndarray:
        self = Hexahedron
        assert n in (8, 20, 26, 27)
        shapes = np.ndarray(n)
        for i in range(n):
            shapes[i] = self.hierarchical_shape(i, x, y, z)
        return shapes

    @staticmethod
    def lagrange_shapes(n, x, y, z) -> np.ndarray:
        self = Hexahedron
        shapes = self.hierarchical_shapes(n, x, y, z)
        if n == 8:
            return shapes
        for a in range(8):
            for b in range(8, 20):
                a_on_b = self.hierarchical_shape(a, self._x[b], self._y[b], self._z[b])
                shapes[a] -= a_on_b * shapes[b]
        if n == 20:
            return shapes
        a_on_b = np.ndarray((6, 20))
        for b in range(20, 26):
            a_on_b[b-20] = self.lagrange_shapes(20, self._x[b], self._y[b], self._z[b])
        for a in range(20):
            for b in range(20, 26):
                shapes[a] -= a_on_b[b-20][a] * shapes[b]
        if n == 26:
            return shapes
        b = 26
        a_on_b = self.lagrange_shapes(26, self._x[b], self._y[b], self._z[b])
        for a in range(26):
            shapes[a] -= a_on_b[a] * shapes[b]
        return shapes


class Pyramid(Element):

    @staticmethod
    def xyz(i: int):
        if i < 5:
            return Hexahedron.xyz(i)
        elif 5 <= i < 9:
            return Hexahedron.xyz(i + 3)
        elif 9 <= i < 13:
            return Hexahedron.xyz(i + 7)
        else:
            assert i == 13
            return Hexahedron.xyz(24)

    @staticmethod
    def hierarchical_shape(i, x, y, z) -> float:
        if i < 4:
            return Hexahedron.hierarchical_shape(i, x, y, z)
        elif i == 4:
            val = Hexahedron.hierarchical_shape(4, x, y, z)
            val += Hexahedron.hierarchical_shape(5, x, y, z)
            val += Hexahedron.hierarchical_shape(6, x, y, z)
            val += Hexahedron.hierarchical_shape(7, x, y, z)
            return val
        elif i == 5:
            val = Hexahedron.hierarchical_shape(8, x, y, z)
            val -= Hexahedron.hierarchical_shape(22, x, y, z) / 4
            return val
        elif i == 6:
            val = Hexahedron.hierarchical_shape(9, x, y, z)
            val -= Hexahedron.hierarchical_shape(21, x, y, z) / 4
            return val
        elif i == 7:
            val = Hexahedron.hierarchical_shape(10, x, y, z)
            val -= Hexahedron.hierarchical_shape(23, x, y, z) / 4
            return val
        elif i == 8:
            val = Hexahedron.hierarchical_shape(11, x, y, z)
            val -= Hexahedron.hierarchical_shape(20, x, y, z) / 4
            return val
        elif i < 13:
            return Hexahedron.hierarchical_shape(i + 7, x, y, z)
        else:
            assert i == 13
            return Hexahedron.hierarchical_shape(24, x, y, z)

    @staticmethod
    def hierarchical_shapes(n, x, y, z) -> np.ndarray:
        self = Pyramid
        assert n in (5, 13, 14)
        shapes = np.ndarray(n)
        for i in range(n):
            shapes[i] = self.hierarchical_shape(i, x, y, z)
        return shapes

    @staticmethod
    def lagrange_shapes(n, x, y, z) -> np.ndarray:
        self = Pyramid
        shapes = self.hierarchical_shapes(n, x, y, z)
        if n == 5:
            return shapes
        for a in range(5):
            for b in range(5, 13):
                x_b, y_b, z_b = self.xyz(b)
                a_on_b = self.hierarchical_shape(a, x_b, y_b, z_b)
                shapes[a] -= a_on_b * shapes[b]
        if n == 13:
            return shapes
        b = 13
        x_b, y_b, z_b = self.xyz(b)
        a_on_b = self.lagrange_shapes(13, x_b, y_b, z_b)
        for a in range(13):
            shapes[a] -= a_on_b[a] * shapes[b]
        return shapes


class TestHexahedron(unittest.TestCase):

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)

    def test_kronecker_delta(self):
        for n in (8, 20, 26, 27):
            for a in range(n):
                x, y, z = Hexahedron.xyz(a)
                shapes = Hexahedron.lagrange_shapes(n, x, y, z)
                self.assertAlmostEqual(1.0, shapes[a])
                self.assertAlmostEqual(1.0, np.linalg.norm(shapes))

    def test_partition_of_unity(self):
        xyz = np.linspace(-1.0, 1.0, 11)
        for n in (8, 20, 26, 27):
            for x in xyz:
                for y in xyz:
                    for z in xyz:
                        shapes = Hexahedron.lagrange_shapes(n, x, y, z)
                        self.assertAlmostEqual(1.0, np.sum(shapes))


class TestPyramid(unittest.TestCase):

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)

    def test_kronecker_delta(self):
        for n in (5, 13, 14):
            for a in range(n):
                x, y, z = Pyramid.xyz(a)
                shapes = Pyramid.lagrange_shapes(n, x, y, z)
                self.assertAlmostEqual(1.0, shapes[a],
                    msg='n = {:0}, a = {:1}'.format(n, a))
                self.assertAlmostEqual(1.0, np.linalg.norm(shapes))

    def test_partition_of_unity(self):
        xyz = np.linspace(-1.0, 1.0, 21)
        for n in (5, 13, 14):
            for x in xyz:
                for y in xyz:
                    for z in xyz:
                        shapes = Pyramid.lagrange_shapes(n, x, y, z)
                        self.assertAlmostEqual(1.0, np.sum(shapes))

    def test_taylor_to_lagrange(self):
        k = 11
        xyz = np.random.rand(k) * 2 - 1
        n_sample = k**3
        degree = 5
        n_taylor = Element.p2n(degree)
        lhs = np.ndarray((n_sample, n_taylor))
        for n_lagrange in (5, 13, 14):
            taylor_to_lagrange = Element.taylor_to_lagrange(Pyramid, n_lagrange, degree)
            self.assertEqual(taylor_to_lagrange.shape, (n_taylor, n_lagrange))
            self.assertEqual(n_lagrange,
                np.linalg.matrix_rank(taylor_to_lagrange))
            rhs = np.ndarray((n_sample, n_lagrange))
            i = 0
            for x in xyz:
                for y in xyz:
                    for z in xyz:
                        lhs[i] = Element.taylor_basis(degree, x, y, z)
                        rhs[i] = Pyramid.lagrange_shapes(n_lagrange, x, y, z)
                        i += 1
            assert i == n_sample
            self.assertAlmostEqual(0,
                np.linalg.norm(lhs @ taylor_to_lagrange - rhs))
            taylor_to_lagrange_lstsq, res, rank, _ = np.linalg.lstsq(lhs, rhs, -1)
            self.assertAlmostEqual(0, np.linalg.norm(res))
            self.assertEqual(rank, n_taylor)
            self.assertEqual(lhs.shape, (n_sample, n_taylor))
            self.assertAlmostEqual(0,
                np.linalg.norm(taylor_to_lagrange - taylor_to_lagrange_lstsq))
            taylor_to_lagrange_csc = sparse.csc_matrix(taylor_to_lagrange)
            print(taylor_to_lagrange_csc.nnz, n_taylor * n_lagrange)
            def test(lhs, matrix, rhs):
                for i in range(1000):
                    diff = lhs @ matrix - rhs
                    assert np.linalg.norm(diff) < 1e-10
            from timeit import default_timer as timer
            start = timer()
            test(lhs, taylor_to_lagrange, rhs)
            end = timer()
            print('t_dense = ', end - start)
            start = timer()
            test(lhs, taylor_to_lagrange_csc, rhs)
            end = timer()
            print('t_csc = ', end - start)


if __name__ == '__main__':
    unittest.main()
