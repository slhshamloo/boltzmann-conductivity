import numpy as np
import scipy.sparse as sp


def solve_cyclic_tridiagonal(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve a cyclic tridiagonal system of equations Ax = b.

    Parameters
    ----------
    A : numpy.ndarray
        Cyclic tridiagonal matrix in diagonal ordered form.
    b : numpy.ndarray
        The right-hand side (can be a vector or matrix).

    Returns
    -------
    numpy.ndarray
        The solution vector x.
    """
    u = np.zeros_like(b)
    free_factor = -A[1, 0] # arbitrary, but avoid division by zero
    A[1, 0] -= free_factor
    A[1, -1] -= A[0, 0] * A[2, -1] / free_factor
    u[0] = 1
    u[-1] = A[0, 0] / free_factor
    # For v, only the dot product is calculated, so only the first
    # and the last elements (the only nonzero elements) are kept
    v_1 = free_factor
    v_n = A[2, -1]

    # Tridiagonal matrix inversions
    # B^{-1}b
    banded_solution = sp.linalg.solve_banded((1, 1), A, b)
    # B^{-1}u
    rank_1_solution = sp.linalg.solve_banded((1, 1), A, u)

    # Dot products
    v_dot_banded_solution = v_1*banded_solution[0] + v_n*banded_solution[-1]
    v_dot_rank_1_solution = v_1*rank_1_solution[0] + v_n*rank_1_solution[-1]

    # A^{-1}b = B^{-1}b - (B^{-1}u) (v^T B^{-1}b) / (1 + v^T B^{-1}u)
    full_solution = (banded_solution - rank_1_solution
                     * (v_dot_banded_solution
                        / (1 + v_dot_rank_1_solution)))

    # Reset the matrix
    A[1, 0] += free_factor
    A[1, -1] += A[0, 0] * A[2, -1] / free_factor

    return full_solution