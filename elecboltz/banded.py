import numpy as np
import scipy


def banded_column(i, j, b, n):
    """
    Get the diagonal ordered form column index
    for a cyclic banded matrix.

    Parameters
    ----------
    i : int
        Row index.
    j : int
        Column index.
    b : int
        Bandwidth.
    n : int
        Size of the matrix.

    Returns
    -------
    int
        The diagonal ordered form column index.
    """
    return b + i - j + (np.abs(i-j)>b) * ((j>i)*n - (i>j)*n)


def solve_cyclic_banded(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""
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
    upper-left, upper-right, lower-left, and lower-right corners of
    :math:`A` respectively, we set the block matrices as
    ..math::
        U_1 = I,\quad U_2 = -C_3 C_1^{-1} V_1 = -C_1,\quad V_2 = C_2.
    
    :math:`U_1` and :math:`V_1` are arbitrary, but they are set such
    that there would not be any divisions by zero in the algorithm.
    """
    bandwidth = (A.shape[0] - 1) // 2
    # construct U and V^T
    I = np.eye(bandwidth)
    U1 = I
    U2 = np.zeros((bandwidth, bandwidth)) # C_3 at first
    V1 = np.zeros((bandwidth, bandwidth)) # same as C_1
    V2 = np.zeros((bandwidth, bandwidth)) # same as C_2
    for ndiag in range(0, bandwidth):
        small_idx = np.arange(0, bandwidth-ndiag)
        large_idx = np.arange(ndiag, bandwidth)
        V1[large_idx, small_idx] = -A[bandwidth+ndiag, :bandwidth-ndiag]
        V1[small_idx, large_idx] = -A[bandwidth-ndiag, ndiag:bandwidth]
        U2[large_idx, small_idx] = A[ndiag, :bandwidth-ndiag]
        V2[small_idx, large_idx] = A[A.shape[0]-ndiag-1, ndiag-bandwidth:]
    U2 = U2 @ np.linalg.inv(V1)
    U = np.vstack((U1, np.zeros((A.shape[1] - 2*bandwidth, bandwidth)), U2))
    # construct B
    B = A.copy()
    for ndiag in range(1, bandwidth+1):
        B[bandwidth+ndiag, -ndiag:] = 0
        B[bandwidth-ndiag, :ndiag] = 0
    idx = np.arange(bandwidth)
    i_idx = idx[:, None]
    j_idx = idx[None, :]
    # U1 @ V1 = V1 since U1 = I
    B[banded_column(i_idx, j_idx, bandwidth, B.shape[1]), j_idx] -= V1
    i_idx = A.shape[1] - bandwidth + idx[:, None]
    j_idx = A.shape[1] - bandwidth + idx[None, :]
    B[banded_column(i_idx, j_idx, bandwidth, B.shape[1]), j_idx] -= (U2 @ V2)[
        i_idx - A.shape[1] + bandwidth, j_idx - A.shape[1] + bandwidth]
    # calculate banded solutions
    BinvU = scipy.linalg.solve_banded((bandwidth, bandwidth), B, U)
    solution = scipy.linalg.solve_banded((bandwidth, bandwidth), B, b)
    # apply the correction
    dense_inversion = np.linalg.inv(
        I + V1@BinvU[:bandwidth] + V2@BinvU[-bandwidth:])
    VdotBinv = V1@solution[:bandwidth] + V2@solution[-bandwidth:]
    solution -= np.linalg.multi_dot((BinvU, dense_inversion, VdotBinv))
    return solution
