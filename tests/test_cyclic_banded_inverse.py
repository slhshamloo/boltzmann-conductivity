import unittest
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.relpath(__file__)) + "/..")
from elecboltz.banded import solve_cyclic_banded, banded_column


class TestCyclicBandedInverse(unittest.TestCase):
    def test_solve_noncyclic_banded(self):
        self.solve_check_random_banded(cyclic=False)

    def test_solve_cyclic_banded(self):
        self.solve_check_random_banded(cyclic=True)
        
    def solve_check_random_banded(self, cyclic=True):
        n = 10
        bandwidth = 2
        A_dense = np.zeros((n, n))
        for ndiag in range(bandwidth + 1):
            small_idx = np.arange(0, n-ndiag)
            large_idx = np.arange(ndiag, n)
            A_dense[small_idx, large_idx] = np.random.random_sample(n - ndiag)
            A_dense[large_idx, small_idx] = np.random.random_sample(n - ndiag)
            if cyclic:
                A_dense[np.arange(ndiag), np.arange(n - ndiag, n)] = \
                    np.random.random_sample(ndiag)
                A_dense[np.arange(n - ndiag, n), np.arange(ndiag)] = \
                    np.random.random_sample(ndiag)
        A_banded = np.zeros((2*bandwidth + 1, n))
        i, j = np.nonzero(A_dense)
        A_banded[banded_column(i, j, bandwidth, n), j] = A_dense[i, j]

        b = np.random.random_sample(n)
        dense_solution = np.linalg.solve(A_dense, b)
        banded_solution = solve_cyclic_banded(A_banded, b)

        for i in range(n):
            self.assertAlmostEqual(
                dense_solution[i], banded_solution[i], places=5,
                msg=f"Solution mismatch at index {i}")


if __name__ == '__main__':
    unittest.main()
