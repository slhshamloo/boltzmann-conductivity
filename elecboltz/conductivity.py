import numpy as np
# units
from scipy.constants import e, hbar, angstrom
# type hinting
from typing import Callable
from collections.abc import Collection
from .bandstructure import BandStructure
from .solve import solve_cyclic_banded


class Conductivity:
    """
    Calculates the conductivity of a material solving the Boltzmann
    transport equation using a finite element method (FEM).

    Parameters
    ----------
    band : BandStructure
        The class holding band structure information of the material.
    field : Collection[float]
        The magnetic field in the x, y, and z directions in units of
        Tesla. Either this or `field_direction` and `field_magnitude`
        must be set.
    field_magnitude : float
        The magnitude of the magnetic field in units of Tesla. Either
        this and `field_direction` or `field` must be set.
    field_direction : Collection[float]
        The unit vector in the direction of the magnetic field. Either
        this and `field_magnitude` or `field` must be set.
    scattering_rate : Callable or float or None
        The (out-)scattering rate as a function of kx, ky, and kz, in
        units of THz. Can also be a constant value instead of a
        function. If None, it will be calculated from the scattering
        kernel.
    scattering_kernel : Callable or None
        The scattering kernel as a function of a pair of coordinates
        (kx, ky, kz) and (kx', ky', kz'), in units of angstrom THz. All
        coordinates are given to the function in order, so the function
        signature would be C(kx, ky, kz, kx', ky', kz'). If None, the
        scattering rate should be specified instead.
    frequency : float
        The frequency of the applied field in units of THz.
        Default is `0.0`.
    
    Attributes
    ----------
    band : BandStructure
        The class holding band structure information of the material.
    field : Collection[float] or None
        The magnetic field in the x, y, and z directions in units of
        Tesla.
    field_magnitude : float
        The magnitude of the magnetic field in units of Tesla.
    field_direction : Collection[float]
        The unit vector in the direction of the magnetic field.
    scattering_rate : Callable or float or Collection[float] or None
        The (out-)scattering rate as a function of kx, ky, and kz. Can
        also be a constant value instead of a function. If initialized
        as None, it will be calculated from the scattering kernel upon
        the next calculation.
    scattering_kernel : Callable or None
        The scattering kernel as a function of a pair of coordinates
        (kx, ky, kz) and (kx', ky', kz'), in units of angstrom THz. All
        coordinates are given to the function in order, so the function
        signature would be C(kx, ky, kz, kx', ky', kz'). If None, the
        scattering rate should be specified instead. The out-scattering
        rate will be calculated from the scattering kernel if the
        scattering rate is not provided.
    frequency : float
        The frequency of the applied field in units of THz.
        If non-zero, the conductivity output will be complex.
    sigma : numpy.ndarray
        The conductivity tensor, which is a 3x3 matrix. Can be
        calculated using the `solve` method. Elements that are not
        calculated yet are set to zero.

    Notes
    -----
    """
    def __init__(self, band: BandStructure, field: Collection[float] = None,
                 field_magnitude: float = 1.0,
                 field_direction: Collection[float] = None,
                 scattering_rate: Callable | float | None = None,
                 scattering_kernel: Callable | None = None,
                 frequency: float = 0.0):
        # avoid triggering setattr in the constructor
        super().__setattr__('band', band)
        super().__setattr__('field_magnitude', field_magnitude)
        super().__setattr__('field_direction', field_direction)
        super().__setattr__('field', field)
        super().__setattr__('scattering_rate', scattering_rate)
        super().__setattr__('scattering_kernel', scattering_kernel)
        super().__setattr__('frequency', frequency)
        self.sigma = np.zeros((3, 3))
        self._saved_solutions = [None, None, None]
        self._velocities = None
        self._vmags = None
        self._vhats = None
        self._velocity_projections = None
        self._bandwidth = None
        self._jacobians = None
        self._jacobian_sums = None
        self._derivative_components = None
        self._derivatives = None
        self._inverse_scattering_length = None
        self._out_scattering = None
        self._derivative_term = None
        self._differential_operator = None
        self._are_elements_saved = False
        self._is_scattering_saved = False
        self._saved_solutions = [None, None, None]

    def __setattr__(self, name, value):
        if name == 'band':
            self.erase_memory()
        if name in ['frequency', 'scattering_rate', 'scattering_kernel']:
            self.erase_memory(elements=False, scattering=True,
                              derivative=False)
        if name in ['field_magnitude', 'field_direction']:
            self.erase_memory(elements=False, scattering=False,
                              derivative=True)
        if name in ['field_magnitude', 'field_direction']:
            if self._derivative_term is not None:
                self._derivative_term *= value / self.field_magnitude
        super().__setattr__(name, value)

    def calculate(self, i: Collection[int] | int | None = None,
                  j: Collection[int] | int | None = None
                  ) -> np.ndarray | float:
        """
        Calculate the conductivity tensor.

        Parameters
        ----------
        i : Collection[int] or int or None, optional
            The index of the first component (row) of the conductivity
            tensor. If None (default), all components are calculated.
        j : Collection[int] or int or None, optional
            The index of the second component (column) of the
            conductivity tensor. If None (default), all components
            are calculated.
        Returns
        -------
        numpy.ndarray or float
            The conductivity tensor component(s) as an ixj matrix.
        """
        if not self._are_elements_saved:
            self._build_elements()
        if self._differential_operator is None:
            self._build_differential_operator()
        
        i, j, j_calc = self._get_calculation_indices(i, j)
        # (A^{-1})^{ij} (v_b)_j
        linear_solution = solve_cyclic_banded(
            self._differential_operator, self._velocity_projections[:, j_calc])
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
        sigma_result = self._velocity_projections[:, i].T @ linear_solution
        sigma_result *= angstrom**2 * e**2 / (4 * np.pi**3 * hbar)

        for idx_row, row in enumerate(i):
            for idx_col, col in enumerate(j):
                self.sigma[row, col] = sigma_result[idx_row, idx_col]
        return sigma_result

    def erase_memory(self, elements: bool = True, scattering: bool = True,
                     derivative: bool = True):
        """
        Erase saved calculations to free memory.

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
            self._bandwidth = None
            self._jacobians = None
            self._jacobian_sums = None
            self._derivative_components = None
            self._derivatives = None
            self._velocity_projections = None
            self._are_elements_saved = False
        if scattering:
            self._inverse_scattering_length = None
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
                self._saved_solutions[col] = []
                j_calc.append(col)
        return i, j, j_calc

    def _build_elements(self):
        """
        Build the arrays corresponding to the discretization of the
        band structure.
        """
        self._velocities = np.column_stack(self.band.velocity_func(
            self.band.kpoints_periodic[:, 0], self.band.kpoints_periodic[:, 1],
            self.band.kpoints_periodic[:, 2]))
        self._vmags = np.linalg.norm(self._velocities, axis=1)
        self._vhats = self._velocities / self._vmags[:, None]
        triangle_coordinates = self.band.kpoints[self.band.kfaces]
        self._calculate_jacobian_sums(triangle_coordinates)
        self._calculate_derivative_sums(triangle_coordinates)
        self._calculate_velocity_projections()
        self._are_elements_saved = True

    def _calculate_jacobian_sums(self, triangle_coordinates):
        """
        Calculate the Jacobian sums for each point and point pair.
        """
        self._jacobians = np.linalg.norm(
            np.cross(triangle_coordinates[:, 1] - triangle_coordinates[:, 0],
                     triangle_coordinates[:, 2] - triangle_coordinates[:, 0]),
            axis=-1)
        # find the bandwidth of the banded matrices, which concerns the
        # "pure", non-periodic neighbors
        self._bandwidth = np.max(np.abs(
            self.band.kfaces - np.roll(self.band.kfaces, 1)))
        # build diagonal ordered matrices of the jacobian sums
        n = len(self.band.kpoints_periodic)
        self._jacobian_sums = np.zeros((2*self._bandwidth + 1, n))
        # convert matrix indices to diagonal ordered form
        # i,j -> bandwidth + i - j, j
        i_idx = self.band.kfaces_periodic[:, :, None]
        j_idx = self.band.kfaces_periodic[:, None, :]
        self._jacobian_sums[(self._bandwidth+i_idx-j_idx) % n, j_idx] += \
            self._jacobians[:, None, None]
    
    def _calculate_derivative_sums(self, triangle_coordinates):
        """
        Calculate the field-independent part of the derivative term.
        """
        n = len(self.band.kpoints_periodic)
        self._derivatives = np.zeros((2*self._bandwidth + 1, n, 3))
        self._derivative_components = (
            triangle_coordinates - np.roll(triangle_coordinates, -2, axis=1))
        i_idx = self.band.kfaces_periodic
        j_idx = np.roll(self.band.kfaces_periodic, -1, axis=1)
        self._derivatives[(self._bandwidth+i_idx-j_idx) % n, j_idx] += \
            self._derivative_components
        self._derivatives[(self._bandwidth+j_idx-i_idx) % n, i_idx] -= \
            self._derivative_components

    def _calculate_velocity_projections(self):
        self._velocity_projections = np.zeros(
            (len(self.band.kpoints_periodic), 3))
        for shift in range(self._bandwidth, -self._bandwidth - 1, -1):
            self._velocity_projections += (
                self._jacobian_sums[self._bandwidth - shift][:, None]
                * np.roll(self._velocities, shift, axis=0)) / 24

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
            self._build_derivative_matrix()
        self._differential_operator = (
            self._out_scattering - e/hbar*angstrom*self._derivative_term)
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
            scattering = self.scattering_rate(self.band.kpoints)
        else:
            scattering = self.scattering_rate
        # separate the optical conductivity case to avoid making
        # the number complex when it is not needed
        if self.frequency == 0.0:
            self._inverse_scattering_length = 1e12 * scattering / self._vmags
        else:
            self._inverse_scattering_length = \
                1e12 * (scattering - 2j*np.pi*self.frequency) / self._vmags
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
        n = len(self.band.kpoints_periodic)
        self._out_scattering = np.zeros((2*self._bandwidth + 1, n))
        for shift in range(self._bandwidth, -self._bandwidth - 1, -1):
            # alpha_{ik} * gamma^k / 60
            self._out_scattering[self._bandwidth] += (
                self._jacobian_sums[self._bandwidth - shift]
                * np.roll(self._inverse_scattering_length, shift)
                ) / 60
            # alpha_{ij} * (gamma^i+gamma^j) / 60
            self._out_scattering[self._bandwidth - shift] += (
                self._jacobian_sums[self._bandwidth - shift]
                * (self._inverse_scattering_length
                   + np.roll(self._inverse_scattering_length, shift))
                ) / 60
        # alpha(i,j,k) * gamma^k / 120
        i_idx = self.band.kfaces_periodic[:, 0]
        j_idx = self.band.kfaces_periodic[:, 1]
        k_idx = self.band.kfaces_periodic[:, 2]
        self._out_scattering[(self._bandwidth+i_idx-j_idx) % n, j_idx] += (
            self._jacobians * self._inverse_scattering_length[k_idx]) / 120
        self._out_scattering[(self._bandwidth+j_idx-i_idx) % n, i_idx] += (
            self._jacobians * self._inverse_scattering_length[k_idx]) / 120
        self._out_scattering[(self._bandwidth+k_idx-i_idx) % n, i_idx] += (
            self._jacobians * self._inverse_scattering_length[j_idx]) / 120
        self._out_scattering[(self._bandwidth+i_idx-k_idx) % n, k_idx] += (
            self._jacobians * self._inverse_scattering_length[j_idx]) / 120
        self._out_scattering[(self._bandwidth+k_idx-j_idx) % n, j_idx] += (
            self._jacobians * self._inverse_scattering_length[i_idx]) / 120
        self._out_scattering[(self._bandwidth+j_idx-k_idx) % n, k_idx] += (
            self._jacobians * self._inverse_scattering_length[i_idx]) / 120

    def _build_derivative_matrix(self):
        """Calculate the derivative matrix (D)"""
        if self.field is None:
            if self.field_direction is None or self.field_magnitude is None:
                raise ValueError(
                    "Either field or field_direction "
                    "and field_magnitude must be set.")
            else:
                self._derivative_term = self.field_magnitude / 6 * np.dot(
                    self._derivatives, self.field_direction)
        else:
            self._derivative_term = np.dot(self._derivatives, self.field) / 6
