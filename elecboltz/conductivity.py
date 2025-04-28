import numpy as np
import scipy.sparse as sp
# units
from scipy.constants import e, hbar, angstrom
# type hinting
from typing import Union
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
    scattering_rate : Union[callable, float, None]
        The (out-)scattering rate as a function of kx, ky, and kz. Can
        also be a constant value instead of a function. If None, it will
        be calculated from the scattering kernel. The default is None.
    scattering_kernel : Union[callable, None]
        The scattering kernel as a function of a pair of coordinates
        (kx, ky, kz) and (kx', ky', kz'). All coordinates are given to
        the function in order, so the function signature would be
        C(kx, ky, kz, kx', ky', kz'). If None, the scattering rate
        should be specified instead. The default is None.
    frequency : float
        The frequency of the applied field. Default is `0.0`.
    
    Attributes
    ----------
    band : BandStructure
        The class holding band structure information of the material.
    field : Collection[float]
        The magnetic field in the x, y, and z directions.
    scattering_rate : Union[callable, float, Collection[float], None]
        The (out-)scattering rate as a function of kx, ky, and kz. Can
        also be a constant value instead of a function. If initialized
        as None, it will be calculated from the scattering kernel upon
        the next calculation.
    scattering_kernel : Union[callable, None]
        The scattering kernel as a function of a pair of coordinates
        (kx, ky, kz) and (kx', ky', kz'). All coordinates are given to
        the function in order, so the function signature would be
        C(kx, ky, kz, kx', ky', kz'). If None, the scattering rate
        should be specified instead. The out-scattering rate will be
        calculated from the scattering kernel if the scattering rate
        is not provided.
    frequency : float
        The frequency of the applied field. If non-zero, the
        conductivity output will be complex.
    sigma : numpy.ndarray
        The conductivity tensor, which is a 3x3 matrix. Can be
        calculated using the `solve` method. Elements that are not
        calculated yet are set to zero.

    Notes
    -----
    """
    def __init__(self, band: BandStructure, field: Collection[float],
                 scattering_rate: Union[callable, float, None] = None,
                 scattering_kernel: Union[callable, None] = None,
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
        self._is_covariant_velocity_saved = [False, False, False]
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ["band", "scattering_rate", "scattering_kernel"]:
            self.erase_memory()
        if name == "frequency" and self._are_terms_saved:
            self.erase_memory(elements=False)
        if name == "field":
            self.field = np.array(value)
            self._is_solution_saved = False

    def solve(self, i: Union[Collection[int], int, None] = None,
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
        if j is None:
            j = range(3)
        sigma_result = np.zeros((len(i), len(j)))
    
        j_calc = []
        for col in j:
            if self._saved_solutions[col] is None:
                self._saved_solutions[col] = []
                j_calc.append(col)

        for layer in range(len(self.band.nlayers)):
            sigma_result += self._calculate_layer_conductivity(
                layer, i, j, j_calc)
        sigma_result *= e**2 / (4 * np.pi**3 * hbar)

        self.sigma[i, j] = sigma_result
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
            lengths and velocities. The default is `True`.
        terms : bool, optional
            If True, erase the differential operator terms and the
            velocity projections (the terms in the final conductivity
            calculation). The default is `True`.
        """
        if elements:
            self._lengths = None
            self._delta_k_hat = None
            self.velocities = None
            self._velocity_magnitudes = None
            self._velocity_hats = None
            self._normals = None
            self._inverse_scattering_length = None
            self._layer_thicknesses = None
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
        return dkz * v.T[:, i] @ linear_solution

    def _generate_elements(self):
        """
        Generate the Finite Elements from the discretization of the
        Fermi surface.
        """
        delta_kx = [np.roll(kx, -1) - kx for kx in self.band.kx]
        delta_ky = [np.roll(ky, -1) - ky for ky in self.band.ky]
        self._lengths = [np.sqrt(dx**2 + dy**2) for dx, dy
                         in zip(delta_kx, delta_ky)]
        self._delta_k_hat = [np.column_stack((dkx, dky)) / k for dkx, dky, k
                             in zip(delta_kx, delta_ky, self._lengths)]
        self.velocities = [
            np.array(self.band.velocity_func(kx, ky, kz)).T for kx, ky, kz
            in zip(self.band.kx, self.band.ky, self.band.kz)]
        self._velocity_magnitudes = [np.linalg.norm(v, axis=-1)
                                     for v in self.velocities]
        self._velocity_hats = [
            v / vmag for v, vmag
            in zip(self.velocities, self._velocity_magnitudes)]
        # Velocity hats over the segments themselves, used for the derivative
        # term. I call them the normals, because they are the normals to each
        # line segment of the discretization (i.e. each element).
        self._normals = [(np.roll(vhat, -1, axis=0) + vhat) / 2
                         for vhat in self._velocity_hats]
        self._discretize_scattering()
        self._layer_thicknesses = ((np.roll(self.band.kz, -1) - self.band.kz
                                   ) % (2*np.pi/self.band.unit_cell[2])
                                   ) / angstrom
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

        if isinstance(self.scattering_rate, callable):
            self._inverse_scattering_length = [
                self.scattering_rate(kx, ky, kz) / v for kx, ky, kz, v
                in zip(self.band.kx, self.band.ky, self.band.kz,
                       self._velocity_magnitudes)]
        elif self.frequency == 0.0:
            self._inverse_scattering_length = [
                self.scattering_rate / self._velocity_magnitudes]
        else:
            self._inverse_scattering_length = [
                (self.scattering_rate - 2j*np.pi*self.frequency)
                / self._velocity_magnitudes]
    
        # TODO: discretize the scattering kernel

    def _calculate_terms(self):
        """
        Calculate the terms needed to make the differential operator
        and calculate the conductivity solving the Boltzmann transport
        equation using an FEM. Note that all matrices are stored in
        diagonal ordered form.
        """
        self._out_scattering = [self._generate_out_scattering(layer)
                                for layer in self.band.nlayers]
        self._velocity_projections = [
            self._generate_velocity_projections(layer)
            for layer in self.band.nlayers]
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
            direction.T @ dkhat for direction, dkhat
            in zip(derivative_directions, self._delta_k_hat)]
        # (vhat x B . dkhat) (delta_{i,j+1} - delta_{i+1,j}) / 2
        derivative_matrix = [dcomp * np.array([-0.5, 0, 0.5]).T
                             for dcomp in derivative_component]        
        self._differential_operator = [
            out_scattering - e/hbar * derivative for out_scattering, derivative
            in zip(self._out_scattering, derivative_matrix)]
        # TODO: also add the in-scattering term

    def _generate_velocity_projections(self, layer):
        """
        Generates the velocity projections v_i for v in every direction
        from velocity vectors v^i and the overlap matrix M
        for a given layer.
        """
        i = np.arange(len(self._lengths))
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
        return overlap @ self.velocities[layer]
    
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
