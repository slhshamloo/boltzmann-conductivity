from .bandstructure import BandStructure

import numpy as np
import scipy.sparse

from typing import Callable, Union
from collections.abc import Sequence

from scipy.constants import e, hbar, angstrom


class Conductivity:
    """
    Calculates the conductivity of a material solving the Boltzmann
    transport equation using a finite element method (FEM).

    Parameters
    ----------
    band : BandStructure
        The class holding band structure information of the material.
    field : Sequence[float]
        The magnetic field in the x, y, and z directions in units of
        Tesla.
    scattering_rate : Callable or float or None
        The (out-)scattering rate as a function of kx, ky, and kz, in
        units of THz. Can also be a constant value instead of a
        function. If None, it will be calculated from the scattering
        kernel.
    scattering_kernel : Callable or None
        The scattering kernel as a function of a pair of coordinates
        ``(kx, ky, kz)`` and ``(kx', ky', kz')``, in units of angstrom
        THz. All coordinates are given to the function in order, so the
        function signature would be ``C(kx, ky, kz, kx', ky', kz')``.
        If None, the scattering rate should be specified instead.
    scattering_params : dict[str, float or Sequence[float]], optional
        Extra parameters passed to the scattering kernel or the
        scattering rate function.
    frequency : float
        The frequency of the applied field in units of THz.
    correct_curvature : bool, optional
        If True, correct for the curvature of the Fermi surface.
    solver : Callable, optional
        The solver used to solve the linear system. Takes the (sparse)
        matrix as the first argument and the right-hand side as the
        second argument. When using a custom solver, keep in mind that
        the right-hand side might not be a vector. So, solvers that
        only work with vectors need to be adapted to solve each column
        of the right-hand side separately.
    
    Attributes
    ----------
    band : BandStructure
        The class holding band structure information of the material.
    field : Sequence[float] or None
        The magnetic field in the x, y, and z directions in units of
        Tesla.
    scattering_rate : Callable or float or Sequence[float] or None
        The (out-)scattering rate as a function of kx, ky, and kz. Can
        also be a constant value instead of a function. If initialized
        as None, it will be calculated from the scattering kernel upon
        the next calculation.
    scattering_kernel : Callable or None
        The scattering kernel as a function of a pair of coordinates
        ``(kx, ky, kz)`` and ``(kx', ky', kz')`` in units of angstrom,
        which returns values in units of angstrom THz. All coordinates
        are given to the function in order, so the function signature
        would be ``C(kx, ky, kz, kx', ky', kz')``. If None, the
        scattering rate should be specified instead. The out-scattering
        rate will be calculated from the scattering kernel if the
        scattering rate is not provided.
    scattering_params : dict[str, float or Sequence[float]], optional
        Extra parameters passed to the scattering kernel or the
        scattering rate function. If, for instance, you also want the
        scattering to depend on the velocities, you can set them here.
    frequency : float
        The frequency of the applied field in units of THz.
        If non-zero, the conductivity output will be complex.
    sigma : numpy.ndarray
        The conductivity tensor, which is a 3 by 3 matrix. Can be
        calculated using the ``solve`` method. Elements that are not
        calculated yet are set to zero.
    correct_curvature : bool
        Whether to correct for the curvature of the Fermi surface.
    solver : Callable
        The solver used to solve the linear system.
    """
    def __init__(
            self, band: BandStructure, field: Sequence[float] = np.zeros(3),
            scattering_rate: Union[Callable, float, None] = None,
            scattering_kernel: Union[Callable, None] = None,
            scattering_params: dict[str, Union[float, Sequence[float]]] = {},
            frequency: float = 0.0, correct_curvature: bool = True,
            solver: Callable = scipy.sparse.linalg.spsolve, **kwargs):
        self.solver = solver
        self.correct_curvature = correct_curvature
        # avoid triggering setattr in the constructor
        super().__setattr__('band', band)
        super().__setattr__('scattering_rate', scattering_rate)
        super().__setattr__('scattering_kernel', scattering_kernel)
        super().__setattr__('scattering_params', scattering_params)
        super().__setattr__('frequency', frequency)
        super().__setattr__('field', np.array(field))
        self._field_magnitude = np.linalg.norm(field)
        if self._field_magnitude != 0:
            self._field_direction = field / self._field_magnitude
        else:
            self._field_direction = np.zeros(3)
        self.sigma = np.zeros((3, 3))
        self._saved_solutions = [None, None, None]
        self._velocities = None
        self._vmags = None
        self._vhats = None
        self._vhat_projections = None
        self._jacobians = None
        self._jacobian_sums = None
        self._derivative_components = None
        self._derivatives = None
        self._scattering_invlen = None
        self._out_scattering = None
        self._derivative_term = None
        self._differential_operator = None
        self._are_elements_saved = False
        self._is_scattering_saved = False
        self._saved_solutions = [None, None, None]

    def __setattr__(self, name, value):
        if name == 'band':
            self.erase_memory()
        if name in ['frequency', 'scattering_rate',
                    'scattering_kernel', 'scattering_params']:
            self.erase_memory(elements=False, scattering=True,
                              derivative=False)
        if name == 'field' and value is not None:
            self.set_field(value)
        super().__setattr__(name, value)
    
    def set_field(self, field):
        field = np.array(field)
        new_magnitude = np.linalg.norm(field)
        if new_magnitude != 0:
            new_direction = field / new_magnitude
        else:
            new_direction = np.zeros(3)
        if np.all(self._field_direction == new_direction):
            if self._derivative_term is not None:
                self._derivative_term *= \
                    new_magnitude / self._field_magnitude
            if self._differential_operator is not None:
                self._differential_operator = \
                    self._out_scattering - e/hbar*self._derivative_term
                self._saved_solutions = [None, None, None]
        else:
            self.erase_memory(elements=False, scattering=False,
                              derivative=True)
        self._field_magnitude = new_magnitude
        self._field_direction = new_direction
        super().__setattr__('field', field)

    def calculate(self, i: Union[Sequence[int], int, None] = None,
                  j: Union[Sequence[int], int, None] = None
                  ) -> Union[np.ndarray, float]:
        """Calculate the conductivity tensor.

        Parameters
        ----------
        i : Sequence[int] or int or None, optional
            The index of the first component (row) of the conductivity
            tensor. If None (default), all components are calculated.
        j : Sequence[int] or int or None, optional
            The index of the second component (column) of the
            conductivity tensor. If None (default), all components
            are calculated.
        Returns
        -------
        numpy.ndarray or float
            The conductivity tensor component(s) as an i by j matrix.
        """
        if not self._are_elements_saved:
            self._build_elements()
        if self._differential_operator is None:
            self._build_differential_operator()
        
        i, j, j_calc = self._get_calculation_indices(i, j)
        # (A^{-1})^{ij} (v_b)_j
        linear_solution = self.solver(self._differential_operator,
                                      self._vhat_projections[:, j_calc])
        if len(linear_solution.shape) == 1:
            linear_solution = linear_solution[:, None]
        # reuse previously calculated solutions
        for col in j:
            if col in j_calc:
                # save solution for potential reuse
                self._saved_solutions[col] = \
                    linear_solution[:, j_calc.index(col)]
            else:
                linear_solution = np.insert(
                    linear_solution, col, self._saved_solutions[col], axis=1)
        # (v_a)_i (A^{-1} v_b)^i
        sigma_result = self._vhat_projections[:, i].T @ linear_solution
        sigma_result *= e**2 / (4 * np.pi**3 * hbar)

        for idx_row, row in enumerate(i):
            for idx_col, col in enumerate(j):
                self.sigma[row, col] = sigma_result[idx_row, idx_col]
        return sigma_result

    def erase_memory(self, elements: bool = True, scattering: bool = True,
                     derivative: bool = True):
        """Erase saved calculations to free memory.

        This class saves already calculated values for the FEM elements
        and matrices to avoid recalculating them every time a new
        element of the conductivity tensor is calculated or a new field
        is applied. This method is provided to erase those values when
        no new calculations are needed and the memory can be freed.

        Parameters
        ----------
        elements : bool, optional
            If True, erase the quantities for each element, like the
            lengths and velocities.
        scattering : bool, optional
            If True, erase the out-scattering and in-scattering terms.
        derivative : bool, optional
            If True, erase the derivative term.
        """
        if elements:
            self._velocities = None
            self._vmags = None
            self._vhats = None
            self._jacobians = None
            self._jacobian_sums = None
            self._derivative_components = None
            self._derivatives = None
            self._vhat_projections = None
            self._are_elements_saved = False
        if scattering:
            self._scattering_invlen = None
            self._out_scattering = None
            self._is_scattering_saved = False
        if derivative:
            self._derivative_term = None
        self._differential_operator = None
        self._saved_solutions = [None, None, None]

    def _get_calculation_indices(self, i, j):
        if i is None:
            i = range(3)
        elif isinstance(i, int):
            i = [i]
        if j is None:
            j = range(3)
        elif isinstance(j, int):
            j = [j]
        j_calc = []
        for col in j:
            if self._saved_solutions[col] is None:
                j_calc.append(col)
        return i, j, j_calc

    def _build_elements(self):
        """
        Build the arrays corresponding to the discretization of the
        band structure.
        """
        self._velocities = np.column_stack(self.band.velocity_func(
            self.band.kpoints[:, 0], self.band.kpoints[:, 1],
            self.band.kpoints[:, 2]))
        self._vmags = np.linalg.norm(self._velocities, axis=1)
        self._vhats = self._velocities / self._vmags[:, None]

        triangle_points = self.band.kpoints[self.band.kfaces] / angstrom
        if self.correct_curvature:
            triangle_points = self._curvature_correct_points(triangle_points)
    
        self._calculate_jacobian_sums(triangle_points)
        self._calculate_derivative_sums(triangle_points)
        self._calculate_velocity_projections()
        self._are_elements_saved = True
    
    def _curvature_correct_points(self, triangle_points):
        """Take the curvature into account for the areas and lengths"""
        kcenters = np.mean(triangle_points, axis=1) * angstrom
        kcenters_tangent = kcenters.copy()
        for _ in range(2):
            kcenters_tangent = self.band._apply_newton_correction(
                kcenters_tangent)
        center_diff = (kcenters_tangent - kcenters) / angstrom
        projected_diff = np.linalg.norm(center_diff, axis=-1)

        point_normals = self._vhats[self.band.kfaces]
        # Turn off warnings for division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # cosine = nhat.cdiff / |cdiff||nhat| and |nhat| = 1
            cosines = np.einsum('ijk,ik->ij', point_normals, center_diff
                                ) / projected_diff[:, None]
            # Handle division by zero
            np.nan_to_num(cosines, copy=False, nan=1.0)
        diff = projected_diff[:, None] / cosines
        return triangle_points + point_normals*diff[:, :, None]

    def _calculate_jacobian_sums(self, triangle_points):
        """Calculate the Jacobian sums for each point and point pair."""
        self._jacobians = np.linalg.norm(
                np.cross(triangle_points[:, 1] - triangle_points[:, 0],
                         triangle_points[:, 2] - triangle_points[:, 0]),
                axis=-1)
        # build diagonal ordered matrices of the jacobian sums
        n = len(self.band.kpoints)
        i_idx = self.band.kfaces[:, 0]
        j_idx = self.band.kfaces[:, 1]
        k_idx = self.band.kfaces[:, 2]
        rows = np.concatenate((i_idx, j_idx, k_idx, i_idx, i_idx,
                                j_idx, j_idx, k_idx, k_idx))
        cols = np.concatenate((i_idx, j_idx, k_idx, j_idx, k_idx,
                                i_idx, k_idx, i_idx, j_idx))
        self._jacobian_sums = scipy.sparse.csr_array(
            (np.tile(self._jacobians, 9), (rows, cols)), shape=(n, n))
        self._jacobian_diagonal = np.zeros(n)
        np.add.at(self._jacobian_diagonal, i_idx, self._jacobians)
        np.add.at(self._jacobian_diagonal, j_idx, self._jacobians)
        np.add.at(self._jacobian_diagonal, k_idx, self._jacobians)

    def _calculate_derivative_sums(self, triangle_points):
        """
        Calculate the field-independent part of the derivative term.
        """
        self._derivative_components = (
            triangle_points - np.roll(triangle_points, -2, axis=1))
        i_idx = self.band.kfaces
        j_idx = np.roll(self.band.kfaces, -1, axis=1)
        k_idx = np.roll(self.band.kfaces, -2, axis=1)
        rows = np.concatenate((i_idx.flat, k_idx.flat))
        cols = np.tile(j_idx.flat, 2)
        self._derivatives = [
            scipy.sparse.csc_array((
            np.tile(component.flat, 2), (rows, cols))) for component in
            self._derivative_components.transpose(2, 0, 1)]
        if self.band.periodic:
            self._derivatives = [
                (self.band.periodic_projector @ derivative
                 @ self.band.periodic_projector.T).tocsc()
                for derivative in self._derivatives]

    def _calculate_velocity_projections(self):
        self._vhat_projections = self._jacobian_sums @ self._vhats / 24
        self._vhat_projections += (
            self._jacobian_diagonal[:, None] * self._vhats / 24)
        if self.band.periodic:
            self._vhat_projections = \
                self.band.periodic_projector @ self._vhat_projections

    def _build_differential_operator(self):
        """
        Build the differential operator from the elements of the
        band structure and the conductivity information.
        """
        if not self._is_scattering_saved:
            self._discretize_scattering()
            self._build_out_scattering_matrix()
            # TODO: calculate the in-scattering matrix
            self._is_scattering_saved = True
        if self._derivative_term is None:
            self._derivative_term = sum(
                Bi / 6 * Di for Bi, Di in zip(self.field, self._derivatives))
        self._differential_operator = (
            self._out_scattering - e/hbar*self._derivative_term)
            # - self._in_scattering_term when implemented

    def _discretize_scattering(self):
        """
        Discretize the scattering rate and the scattering kernel
        for each element.
        """
        if self.scattering_rate is None:
            if self.scattering_kernel is None:
                raise ValueError(
                    "Either scattering_rate or scattering_kernel must be set.")
            else:
                self._calculate_out_scattering_from_kernel()

        if isinstance(self.scattering_rate, Callable):
            scattering = self.scattering_rate(
                self.band.kpoints[:, 0], self.band.kpoints[:, 1],
                self.band.kpoints[:, 2], **self.scattering_params)
        else:
            scattering = self.scattering_rate
        # separate the optical conductivity case to avoid making
        # the number complex when it is not needed
        if self.frequency == 0.0:
            self._scattering_invlen = 1e12 * scattering / self._vmags
        else:
            self._scattering_invlen = \
                1e12 * (scattering - 2j*np.pi*self.frequency) / self._vmags
        # scattering_invlen is the inverse scattering length gamma
        # TODO: discretize the scattering kernel

    def _calculate_out_scattering_from_kernel(self):
        """
        Calculate the scattering rate by integrating over
        the scattering kernel.
        """
        # TODO: implement this
        pass

    def _build_out_scattering_matrix(self):
        """Calculate the out-scattering matrix (Gamma)"""
        self._out_scattering = (
            # alpha_ij * gamma^i
            (self._jacobian_sums * self._scattering_invlen[:, None]).tocsc()
            # alpha_ij * gamma^j
            + (self._jacobian_sums * self._scattering_invlen[None, :]).tocsc()
            # sum_k alpha_ik * gamma^k
            + scipy.sparse.diags_array(
                self._jacobian_sums @ self._scattering_invlen, format='csc')
            ) / 60
        # alpha(i,j,k) * gamma^k / 120
        i_idx = self.band.kfaces[:, 0]
        j_idx = self.band.kfaces[:, 1]
        k_idx = self.band.kfaces[:, 2]
        n = len(self.band.kpoints)
        rows = np.concatenate((i_idx, i_idx, j_idx, j_idx, k_idx, k_idx))
        cols = np.concatenate((j_idx, k_idx, i_idx, k_idx, i_idx, j_idx))
        data = np.concatenate((
            self._jacobians * self._scattering_invlen[k_idx],
            self._jacobians * self._scattering_invlen[j_idx],
            self._jacobians * self._scattering_invlen[k_idx],
            self._jacobians * self._scattering_invlen[i_idx],
            self._jacobians * self._scattering_invlen[j_idx],
            self._jacobians * self._scattering_invlen[i_idx])) / 120
        self._out_scattering += scipy.sparse.csc_array(
            (data, (rows, cols)), shape=(n, n))
        if self.band.periodic:
            self._out_scattering = (
                self.band.periodic_projector @ self._out_scattering
                @ self.band.periodic_projector.T).tocsc()
