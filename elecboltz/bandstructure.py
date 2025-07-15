from .integrate import adaptive_octree_integrate

import numpy as np
import sympy
import scipy.sparse
from skimage.measure import marching_cubes

from typing import Union
from collections.abc import Sequence

from scipy.constants import hbar, eV, angstrom
# conversion from energy gradient units to m/s for velocity
velocity_units = 1e-3 * eV * angstrom / hbar


class BandStructure:
    """Contains bandstructure information for a given material.

    In addition to the dispersion relation and general parameters, this
    class also contains methods for discretizing the Fermi surface and
    calculating electronic properties.

    Parameters
    ----------
    dispersion : str
        The dispersion relation. Expresses the dispersion relation
        in terms of symbols in ``wavevector_names`` and additional
        parameters in ``band_params``. It must be parsable and
        differentiable by ``sympy``. Energy units are milli eV.
    chemical_potential : float
        The chemical potential in milli eV.
    unit_cell : Sequence[float]
        The dimensions of the unit cell in angstrom.
    band_params : dict, optional
        The parameters of the dispersion relation. Energy units are
        milli eV and distance units are angstrom.
    domain_size : Sequence[float]
        The ratio of the reciprocal space domain sidelengths to simple
        cubic unit cell dimensions in reciprocal space. The product
        of the numbers in this collection must be equal to the number
        of atoms in the conventional unit cell specified by
        ``unit_cell``.
    periodic : bool or int or Sequence[bool] or Sequence[int]
        If bool, whether periodic boundary conditions are applied to all
        axes or not. If a single int, specifies which single axis is 
        periodic. If a collection, specifies which axes are periodic.
        If the collection is of integers, the integers specify the
        periodic axes, e.g. ``[0, 2]`` means periodic in `x` and `z`
        axes. If the collection is of booleans, it specifies whether
        each axis is periodic or not, e.g. ``[True, False, True]`` means
        periodic in `x` and `z` axes, but not in `y` axis.
    axis_names : str or Sequence[str], optional
        The names of the unit cell axes. Must be parsable by
        `sympy.symbols`.
    wavevector_names : str or Sequence[str], optional
        The names of the wavevector components. Must be parsable by
        `sympy.symbols`.
    resolution :  int or Sequence[int], optional
        Controls the resolution of the grids used for discretizing the
        Fermi surface. If a collection of integers is provided, each
        element corresponds to the resolution along the respective
        axis. If a single integer is provided, it is used for all axes.
    ncorrect : int, optional
        The number of correction steps for improving the accuracy of
        the discretization of the Fermi surface.
    sort_axis : int, optional
        The axis along which to sort the points after triangulation.
        If None, do not sort the points.

    Attributes
    ----------
    dispersion : str
        The dispersion relation. Updating this will automatically
        update ``energy_func`` and ``velocity_func``.
    chemical_potential : float
        The chemical potential in milli eV.
    unit_cell : Sequence[float]
        The dimensions of the unit cell in angstrom.
    domain_size : Sequence[float]
        The ratio of the reciprocal space domain sidelengths to simple
        cubic unit cell dimensions in reciprocal space. The product
        of the numbers in this collection must be equal to the number
        of atoms in the conventional unit cell specified by
        ``unit_cell``.
    periodic : Sequence[int]
        The periodic axes.
    band_params : dict
        The parameters of the dispersion relation. Energy units are
        milli eV and distance units are angstrom.
    kpoints : (N, 3) numpy.ndarray
        The discretized k-points on the Fermi surface. Each row
        corresponds to a k-point in the form ``[kx, ky, kz]``.
    kfaces : (F, 3) numpy.ndarray
        The faces of the triangulated surface in k-space. Each row
        corresponds to a face in the form ``[i, j, k]``, where
        ``i``, ``j``, and ``k`` are the indices of the vertices of
        the face in the ``kpoints`` array.
    periodic_projector : scipy.sparse.csr_array
        Projects quantities into the periodic k-space, where points that
        are periodic images of each other are mapped to the same point.
    resolution : int or Sequence[int]
        The resolution of the grids used for approximating the Fermi
        surface geometry with the marching cubes algorithm.
    ncorrect : int
        The number of Newton--Raphson steps applied to correct the
        triangulated surface after the marching cubes algorithm.
    sort_axis : int or None
        The axis along which to sort the points after triangulation.
    axis_names : str or Sequence[str]
        The names of the unit cell axes.
    wavevector_names : str or Sequence[str]
        The names of the wavevector components.
    """
    def __init__(
            self, dispersion: str, chemical_potential: float,
            unit_cell: Sequence[float], band_params: dict = {},
            domain_size: Sequence[float] = [1.0, 1.0, 1.0],
            periodic: Union[bool, Sequence[Union[int, bool]]] = True,
            axis_names: Union[Sequence[str], str] = ['a', 'b', 'c'],
            wavevector_names: Union[Sequence[str], str] = ['kx', 'ky', 'kz'],
            resolution: Union[int, Sequence[int]] = 21, n_correct: int = 2,
            sort_axis: int = None, **kwargs):
        # avoid triggering the __setattr__ method for the first time
        super().__setattr__('dispersion', dispersion)
        self.band_params = band_params
        self.chemical_potential = chemical_potential
        self.unit_cell = unit_cell
        self.domain_size = domain_size
        self.periodic = periodic
        self.axis_names = axis_names
        self.wavevector_names = wavevector_names
        self.resolution = resolution
        self.n_correct = n_correct
        self._parse_dispersion()
        self.kpoints = None
        self.kfaces = None
        self.periodic_projector = None
        self.sort_axis = sort_axis

    def __setattr__(self, name, value):
        if name == 'dispersion':
            self._parse_dispersion()
        if name == 'resolution':
            if isinstance(value, Sequence):
                value = np.array(value)
            else:
                value = np.array([value, value, value])
        if name in {'unit_cell', 'domain_size'}:
            value = np.array(value, dtype=float)
        if name == 'periodic':
            if isinstance(value, bool):
                if value:
                    value = [0, 1, 2]
                else:
                    value = []
            if isinstance(value, int):
                value = [value]
            elif isinstance(value, Sequence) and all(
                    isinstance(i, bool) for i in value):
                value = [i for i, v in enumerate(value) if v]
        super().__setattr__(name, value)
    
    def __getstate__(self):
        """Get the state of the object for pickling."""
        state = self.__dict__.copy()
        # remove the full energy and velocity functions
        state['_energy_func_full'] = None
        state['_velocity_funcs_full'] = None
        return state
    
    def __setstate__(self, state):
        """Set the state of the object after unpickling."""
        self.__dict__.update(state)
        # re-parse the dispersion relation to restore the full functions
        self._parse_dispersion()
    
    def discretize(self):
        """Discretize the Fermi surface.

        First, the surface is triangulated using the marching cubes
        algorithm with ``resolution`` controlling the resolution of the
        grid. Next, to improve the accuracy of the isosurface,
        ``n_correct`` steps of the Newton--Raphson root-finding method
        are applied to the output of marching cubes. Finally, after the
        surface construction, periodic boundary conditions are applied
        to "stitch" the open ends of the surface together.
        """
        self._gvec = self.domain_size * np.pi / self.unit_cell
        self.kpoints, self.kfaces, _, _ = marching_cubes(
            self.energy_func(*np.mgrid[
                -self._gvec[0]:self._gvec[0]:1j*self.resolution[0],
                -self._gvec[1]:self._gvec[1]:1j*self.resolution[1],
                -self._gvec[2]:self._gvec[2]:1j*self.resolution[2]]),
            level=self.chemical_potential)
        self.kpoints *= (2*self._gvec / (self.resolution-1))[None, :]
        self.kpoints -= self._gvec[None, :]
        for _ in range(self.n_correct):
            self.kpoints = self._apply_newton_correction(self.kpoints)
        if self.sort_axis:
            self._sort_and_reindex(self.sort_axis)
        self._stitch_periodic_boundaries()

    def calculate_filling_fraction(self, depth: int = 7) -> float:
        """Calculate the filling fraction n of the material.

        The filling fraction is calculated by integrating the volume
        in the reciprocal space having energy bellow the Fermi level
        (calculated by an adaptive octree integration method), then
        dividing by the volume of the unit cell in the reciprocal space.

        Parameters
        ----------
        depth : int, optional
            The depth of the adaptive octree integration. Higher values
            result in more accurate integration, but take exponentially
            longer to compute.

        Returns
        -------
        float
            The filling fraction n of the material.
        """
        self._gvec = self.domain_size * np.pi / self.unit_cell
        # the extra factor of 2 is the spin degeneracy
        return 2 * adaptive_octree_integrate(
            lambda kx, ky, kz: (self.energy_func(kx, ky, kz)
                                < self.chemical_potential),
            (-self._gvec[0], self._gvec[0], -self._gvec[1], self._gvec[1],
             -self._gvec[2], self._gvec[2]), depth=depth
            ) / 8 / np.prod(self._gvec)

    def calculate_electron_density(self, depth: int = 7) -> float:
        """Calculate the electron density n_e of the material.

        Note that the surface needs to be discretized before calling
        this method. First, the filling fraction is calculated (see
        ``calculate_filling_fraction``). The electron density is
        obtained by dividing the filling fraction by the volume of
        the unit cell in real space.

        Parameters
        ----------
        depth : int, optional
            The depth of the adaptive octree integration in
            ``calculate_filling_fraction``.

        Returns
        -------
        float
            The electron density n_e of the material in SI units.
        """
        filling_fraction = self.calculate_filling_fraction(depth)
        # the volume of the unit cell in real space is scaled by
        # the inverse scaling of the unit cell in reciprocal space
        unit_cell_volume = (np.prod(self.unit_cell) * angstrom**3
                            / np.prod(self.domain_size))
        return filling_fraction / unit_cell_volume

    def calculate_mass(self):
        """Calculate the effective mass of the charge carries.

        Returns
        -------
        float
            The effective mass divided by the rest mass of
            the electron, m_e.
        """
        # Placeholder for actual calculation
        return 0.0
    
    def energy_func(self, kx, ky, kz):
        """Calculate the energy at the given k-point.

        Parameters
        ----------
        kx, ky, kz : float
            The components of the wavevector in angstrom^-1.

        Returns
        -------
        object like kx, ky, kz
            The energy at the given k-point in milli eV.
        """
        return self._energy_func_full(
            kx, ky, kz, *self.unit_cell, **self.band_params)
    
    def velocity_func(self, kx, ky, kz):
        """Calculate the velocity at the given k-point.

        Parameters
        ----------
        kx, ky, kz : float
            The components of the wavevector in angstrom^-1.

        Returns
        -------
        list of 3 objects like kx, ky, kz
            The velocity vector at the given k-point in m/s.
        """
        return [vfunc(kx, ky, kz, *self.unit_cell, **self.band_params)
                for vfunc in self._velocity_funcs_full]

    def _parse_dispersion(self):
        """
        Parse the dispersion relation and extract the necessary
        information for further calculations.
        """
        ksymbols = sympy.symbols(self.wavevector_names)
        all_symbols = (ksymbols + sympy.symbols(self.axis_names)
                       + sympy.symbols(list(self.band_params.keys())))
        # symbolic expressions
        self._energy_sympy = sympy.sympify(self.dispersion)
        self._velocities_sympy = [
            sympy.diff(self._energy_sympy, k) * velocity_units
            for k in sympy.symbols(self.wavevector_names)]
        # replace zero velocities with zero arrays
        for i, v in enumerate(self._velocities_sympy):
            if v == 0:
                self._velocities_sympy[i] = f"numpy.zeros_like({ksymbols[i]})"
        # convert expressions into python functions
        self._energy_func_full = sympy.lambdify(
            all_symbols, self._energy_sympy)
        self._velocity_funcs_full = [
            sympy.lambdify(all_symbols, vexpr, 'numpy')
            for vexpr in self._velocities_sympy]

    def _sort_and_reindex(self, sort_axis):
        new_order, self.kfaces = self._generate_reindex(sort_axis)
        self.kpoints = self.kpoints[new_order]
    
    def _generate_reindex(self, sort_axis):
        new_order = np.argsort(self.kpoints[:, sort_axis])
        old_to_new_map = np.empty(len(new_order), dtype=int)
        old_to_new_map[new_order] = np.arange(len(new_order))
        return new_order, old_to_new_map[self.kfaces]

    def _stitch_periodic_boundaries(self):
        """
        Find duplicate points on the periodic boundaries, then make the
        periodic mesh arrays.
        """
        duplicates = dict()
        threshold = np.min(self._gvec / self.resolution) / 10
        for axis in self.periodic:
            low_border = np.argwhere(
                self.kpoints[:, axis] + self._gvec[axis] < threshold).ravel()
            high_border = np.argwhere(
                self.kpoints[:, axis] - self._gvec[axis] > -threshold).ravel()
            if len(low_border) == 0 or len(high_border) == 0:
                continue

            min_dist = min(self._get_min_border_distance(low_border),
                           self._get_min_border_distance(high_border))

            k1 = self.kpoints[low_border][None, :]
            k2 = self.kpoints[high_border][:, None]
            kdiff = k2 - k1
            kdiff[:, :, axis] += self._gvec[axis]
            kdiff[:, :, axis] %= 2 * self._gvec[axis]
            kdiff[:, :, axis] -= self._gvec[axis]
            kdiff = np.linalg.norm(kdiff, axis=-1)

            min_pair = np.argmin(kdiff, axis=1)
            is_duplicate = kdiff[np.arange(len(high_border)), min_pair
                                 ] < min_dist / 2
            duplicates.update(dict(zip(
                high_border[is_duplicate],
                low_border[min_pair[is_duplicate]])))
        self._build_periodic_projector(duplicates)
    
    def _get_min_border_distance(self, border):
        """
        Find minimum intra-layer distance to set the threshold
        for duplicate point detection.
        """
        is_triangle_point_in_border = np.isin(self.kfaces, border)
        border_triangles = self.kfaces[np.any(
            is_triangle_point_in_border, axis=1)]
        points = self.kpoints[border_triangles]
        is_triangle_point_in_border = is_triangle_point_in_border[
            np.any(is_triangle_point_in_border, axis=1)]
        is_pair_intra_layer = np.logical_xor(
            is_triangle_point_in_border,
            np.roll(is_triangle_point_in_border, 1, axis=-1))
        return np.min(np.linalg.norm(
            (points - np.roll(points, 1, axis=1))[is_pair_intra_layer],
            axis=-1))

    def _build_periodic_projector(self, duplicates):
        """
        Build the periodic kpoints and kfaces arrays by removing
        duplicate points and reindexing.
        """
        if not duplicates:
            self.periodic_projector = scipy.sparse.eye(
                len(self.kpoints), format='csr')
        else:
            unique_mask = np.full(len(self.kpoints), True)
            unique_mask[list(duplicates.keys())] = False
            reindex_map = np.cumsum(unique_mask) - 1
            reindex_map[list(duplicates.keys())] = reindex_map[
                list(duplicates.values())]
            self.periodic_projector = scipy.sparse.csr_array(
                (np.ones(len(self.kpoints)),
                (reindex_map, np.arange(len(self.kpoints)))),
                shape=(np.count_nonzero(unique_mask), len(self.kpoints)))

    def _apply_newton_correction(self, points):
        residuals = self.energy_func(
            points[:, 0], points[:, 1], points[:, 2]) - self.chemical_potential
        gradients = np.column_stack(self.velocity_func(
            points[:, 0], points[:, 1], points[:, 2])) / velocity_units
        gradient_norms = np.linalg.norm(gradients, axis=-1)
        return points - (residuals/gradient_norms**2)[:, None]*gradients
