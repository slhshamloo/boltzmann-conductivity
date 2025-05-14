import numpy as np
import scipy.sparse as sp
import opt_einsum
# units
from scipy.constants import e, hbar, angstrom
# type hinting
from typing import Union, Callable
from collections.abc import Collection
from .bandstructure import BandStructure
from .solve import solve_cyclic_tridiagonal

class Conductivity:
    """
    Calculates the conductivity of a material solving the Boltzmann
    transport equation using a finite element method (FEM).

    Parameters
    ----------
    band : BandStructure
        The class holding band structure information of the material.
    field : Collection[float]
        The magnetic field in the x, y, and z directions.
    scattering_rate : Union[Callable, float, None]
        The (out-)scattering rate as a function of kx, ky, and kz, in
        units of THz. Can also be a constant value instead of a
        function. If None, it will be calculated from the scattering
        kernel.
    scattering_kernel : Union[Callable, None]
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
    field : Collection[float]
        The magnetic field in the x, y, and z directions.
    scattering_rate : Union[Callable, float, Collection[float], None]
        The (out-)scattering rate as a function of kx, ky, and kz. Can
        also be a constant value instead of a function. If initialized
        as None, it will be calculated from the scattering kernel upon
        the next calculation.
    scattering_kernel : Union[Callable, None]
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
    def __init__(self, band: BandStructure, field: Collection[float],
                 scattering_rate: Union[Callable, float, None] = None,
                 scattering_kernel: Union[Callable, None] = None,
                 frequency: float = 0.0):
        self.band = band
        self.field = field
        self.scattering_rate = scattering_rate
        self.scattering_kernel = scattering_kernel
        self.frequency = frequency
        self.sigma = np.zeros((3, 3))
        self._saved_solutions = [None, None, None]
        self._are_elements_saved = False
        self._are_terms_saved = False
        self._is_solution_saved = False
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ["band", "scattering_rate", "scattering_kernel"]:
            self.erase_memory()
        if name == "frequency" and self._are_terms_saved:
            self.erase_memory(elements=False)
        if name == "field":
            super().__setattr__(name, np.array(value))
            self._is_solution_saved = False

    def calculate(self, i: Union[Collection[int], int, None] = None,
                  j: Union[Collection[int], int, None] = None
                  ) -> Union[np.ndarray, float]:
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
            self._generate_elements()
        if not self._are_terms_saved:
            self._calculate_terms()
        if not self._is_solution_saved:
            self._generate_differential_operator()
        
        if i is None:
            i = range(3)
        elif isinstance(i, int):
            i = [i]
        if j is None:
            j = range(3)
        elif isinstance(j, int):
            j = [j]
        sigma_result = np.zeros((len(i), len(j)))
    
        j_calc = []
        for col in j:
            if self._saved_solutions[col] is None:
                self._saved_solutions[col] = []
                j_calc.append(col)

        for layer in range(self.band.nlayers):
            sigma_result += self._calculate_layer_conductivity(
                layer, i, j, j_calc)
        sigma_result *= e**2 / (4 * np.pi**3 * hbar)

        for idx_row, row in enumerate(i):
            for idx_col, col in enumerate(j):
                self.sigma[row, col] = sigma_result[idx_row, idx_col]
        return sigma_result

    def erase_memory(self, elements: bool = True, terms: bool = True):
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
        terms : bool, optional
            If True, erase the differential operator terms and the
            velocity projections (the terms in the final conductivity
            calculation).
        """
        if elements:
            self._velocities = None
            self._vmags = None
            self._vhats = None
            self._jacobians = None
            self._jacobian_sums = None
            self._derivative_components = None
            self._derivatives = None
            self._are_elements_saved = False
        if terms:
            self._overlap = None
            self._out_scattering = None
            self._are_terms_saved = False
        self._differential_operator = None
        self._saved_solutions = [None, None, None]
        self._is_solution_saved = False
    
    def _calculate_out_scattering_from_kernel(self):
        """
        Calculate the scattering rate by integrating over
        the scattering kernel.
        """
        # TODO: implement this
        pass

    def _calculate_layer_conductivity(self, layer, i, j, j_calc):
        """Calculate the conductivity for a single layer"""
        A = self._differential_operator[layer]
        v = self._velocity_projections[layer]
        dkz = self._layer_thicknesses[layer]
        # (A^{-1})^{ij} (v_b)_j
        linear_solution = solve_cyclic_tridiagonal(A, v[:, j_calc])
        # reuse previously calculated solutions
        for col in j:
            if col in j_calc:
                # save solution for potential reuse
                self._saved_solutions[col].append(
                    linear_solution[:, j_calc.index(col)])
            else:
                linear_solution = np.insert(
                    linear_solution, col,
                    self._saved_solutions[col][layer], axis=1)
        # (v_a)_i (A^{-1} v_b)^i
        return dkz * v[:, i].T @ linear_solution

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
        self._discretize_scattering()
        triangle_coordinates = self.band.kpoints[self.band.kfaces]
        self._calculate_jacobian_sums(triangle_coordinates)
        self._calculate_derivative_sums(triangle_coordinates)
        self._are_elements_saved = True

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
                self.scattering_rate = self._generate_out_scattering()

        if isinstance(self.scattering_rate, Callable):
            self._inverse_scattering_length = (
                1e12 * self.scattering_rate(self.band.kpoints)
                / self.band.velocities)
        elif self.frequency == 0.0:
            self._inverse_scattering_length = (
                1e12 * self.scattering_rate(self.band.kpoints)
                / self.band.velocities)
        else:
            self._inverse_scattering_length = (
                1e12 * (self.scattering_rate(self.band.kpoints)
                        - 2j*np.pi*self.frequency) / self.band.velocities)
        # TODO: discretize the scattering kernel
    
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
        self._jacobian_sums = np.zeros(
            (2*self._bandwidth + 1, len(self.band.kpoints_periodic)))
        # convert matrix indices to diagonal ordered form
        # i,j -> bandwidth + i - j, j
        i_idx = self.band.kfaces_periodic[:, :, None]
        j_idx = self.band.kfaces_periodic[:, None, :]
        self._jacobian_sums[self._bandwidth + i_idx - j_idx, j_idx] += \
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

    def _calculate_terms(self):
        """
        Calculate the terms needed to make the differential operator
        and calculate the conductivity solving the Boltzmann transport
        equation using an FEM. Note that all matrices are stored in
        diagonal ordered form.
        """
        self._out_scattering = [self._generate_out_scattering(layer)
                                for layer in range(self.band.nlayers)]
        self._velocity_projections = [
            self._generate_velocity_projections(layer)
            for layer in range(self.band.nlayers)]
        # TODO: calculate the in-scattering matrix
        self._are_terms_saved = True
    
    def _generate_differential_operator(self):
        """
        Generates the differential operator form the comprising
        matrices and the normals.
        """
        # vhat x B
        derivative_directions = [
            np.cross(nhat, self.field) for nhat in self._normals]
        # vhat x B . dkhat
        derivative_component = [
            opt_einsum.contract('ij,ij->i', u, dkhat)
            for u, dkhat in zip(derivative_directions, self._delta_k_hat)]
        # (vhat x B . dkhat) (delta_{i,j+1} - delta_{i+1,j}) / 2
        derivative_matrix = [dcomp[None, :] * np.array([[0.5], [0.0], [-0.5]])
                             for dcomp in derivative_component]
        self._differential_operator = [
            out_scattering - e/hbar*derivative for out_scattering, derivative
            in zip(self._out_scattering, derivative_matrix)]
        # TODO: also add the in-scattering term

    def _generate_velocity_projections(self, layer):
        """
        Generates the velocity projections v_i for v in every direction
        from velocity vectors v^i and the overlap matrix M
        for a given layer.
        """
        i = np.arange(len(self._lengths[layer]))
        i_plus_one = np.roll(i, -1)
        lengths = self._lengths[layer]
        # (l_i + l_{i+1} / 3) delta_{ij}
        overlap_main = sp.csr_matrix(
            ((lengths + np.roll(lengths, 1)) / 3, (i, i)))
        # (l_i / 6) delta_{i+1,j}
        overlap_upper = sp.csr_matrix((lengths / 6, (i_plus_one, i)))
        # (l_i / 6) delta_{i,j+1}
        overlap_lower = sp.csr_matrix((lengths / 6, (i, i_plus_one)))
        overlap = overlap_main + overlap_upper + overlap_lower
        return overlap @ self._velocity_hats[layer]
    
    def _generate_out_scattering(self, layer):
        """Generates the out-scattering (Gamma) matrix for a given layer."""
        lengths = self._lengths[layer]
        gamma = self._inverse_scattering_length[layer]
        gamma_plus_one = np.roll(gamma, -1)
        gamma_minus_one = np.roll(gamma, 1)
        # Gamma matrix
        minor_diagonal_term = lengths / 12 * (gamma + gamma_plus_one)
        major_diagonal_term = (
            (np.roll(lengths, 1) + lengths) * gamma / 4
            + lengths * gamma_plus_one / 12
            + np.roll(lengths, 1) * gamma_minus_one / 12)
        return np.vstack(
            [minor_diagonal_term, major_diagonal_term, minor_diagonal_term])
