import numpy as np
import scipy


def solve_cyclic_banded(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system of equations Ax = b
    with A being a cyclic banded matrix.

    Parameters
    ----------
    A : numpy.ndarray
        Cyclic banded matrix in diagonal ordered form. It is assumed
        that the number of upper and lower diagonals is the same.
    b : numpy.ndarray
        The right-hand side (can be a vector or matrix).

    Returns
    -------
    numpy.ndarray
        The solution vector x.
    
    Notes
    -----
    This uses a combination of the banded matrix solver in scipy (which
    uses the LAPACK routine dgbsv) and the Sherman--Morrison--Woodbury
    formula to handle the "corner" terms (the terms coming from the
    cyclic nature of the matrix).

    The matrix :math:`A` can be expressed as the sum of a banded matrix
    :math:`B` and a low-rank update :math:`UV^T`. Then, using the
    Sherman--Morrison--Woodbury formula, we can express the inverse as
    .. math::
        A^{-1} = B^{-1} - B^{-1}U (I + V^TB^{-1}U)^{-1} V^TB^{-1}.
    
    where :math:`I` is the identity matrix with the correct dimensions.
    To do this in practice, first we define the matrices as
    .. math::
        B = A - UV^T,\quad
        U = \begin{pmatrix}
            U_1 \\ 0 \\ U_2
        \end{pmatrix},\quad
        V^T = \begin{pmatrix}
            V_1 & 0 & V_2
        \end{pmatrix}.

    The block matrices used to define :math:`U` and :math:`V` are not
    well constrained by just the cyclic terms. So, if we define
    :math:`C_1`, :math:`C_2`, :math:`C_3`, and :math:`C_4` as the
    upper-right, upper-left, lower-left, and lower-right corners of
    :math:`A` respectively, we set the block matrices as
    ..math::
        U_1 = I,\quad U_2 = -C_3 C_1^{-1} V_1 = -C_1,\quad V_2 = C_2.
    
    :math:`U_1` and :math:`V_1` are arbitrary, but they are set such
    that there would not be any divisions by zero in the algorithm.
    """
    # useful variables
    bandwidth = (A.shape[0] - 1) // 2
    idx = np.arange(bandwidth)
    # construct U and V^T
    I = np.eye(bandwidth)
    U1 = I
    U2 = np.zeros((bandwidth, bandwidth)) # C_3 at first
    V1 = np.zeros((bandwidth, bandwidth)) # same as C_1
    V2 = np.zeros((bandwidth, bandwidth)) # same as C_2
    for ndiag in range(0, bandwidth):
        lower_diag = idx[:bandwidth-ndiag]
        upper_diag = idx[ndiag-bandwidth:]
        V1[lower_diag, lower_diag] = -A[bandwidth+ndiag, :bandwidth-ndiag]
        V1[upper_diag, upper_diag] = -A[bandwidth-ndiag, ndiag:bandwidth]
        U2[lower_diag, lower_diag] = A[ndiag, :bandwidth-ndiag]
        V2[upper_diag, upper_diag] = A[A.shape[0]-ndiag-1, ndiag-bandwidth:]
    U2 = U2 @ np.linalg.inv(V1)
    # construct B
    B = A.copy()
    i_idx = idx[:, None]
    j_idx = idx[None, :]
    B[bandwidth + i_idx - j_idx, j_idx] -= V1 # U1 @ V1 but U1 is I
    i_idx = A.shape[1] - 1 - idx[:, None]
    j_idx = A.shape[1] - 1 - idx[None, :]
    B[bandwidth + i_idx - j_idx, j_idx] -= U2 @ V2
    # calculate banded solutions
    BinvU1 = scipy.linalg.solve_banded((bandwidth, bandwidth),
                                       A[:, :bandwidth], U1)
    BinvU2 = scipy.linalg.solve_banded((bandwidth, bandwidth),
                                       A[:, -bandwidth:], U2)
    banded_solution = scipy.linalg.solve_banded((bandwidth, bandwidth), B, b)
    # apply the correction
    dense_inversion = np.linalg.inv(I + V1 @ BinvU1 + V2 @ BinvU2)
    solution = banded_solution.copy()
    solution[:bandwidth, :] -= (BinvU1 @ dense_inversion @ V1
                             @ banded_solution[:bandwidth, :])
    solution[:bandwidth, :] -= (BinvU1 @ dense_inversion @ V2
                                @ banded_solution[-bandwidth:, :])
    solution[-bandwidth:, :] -= (BinvU2 @ dense_inversion @ V1
                                 @ banded_solution[:bandwidth, :])
    solution[-bandwidth:, :] -= (BinvU2 @ dense_inversion @ V2
                                 @ banded_solution[-bandwidth:, :])
    return solution
