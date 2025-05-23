import numpy as np
import sympy
import pymeshlab
from skimage.measure import marching_cubes
# units
from scipy.constants import hbar, eV, angstrom
# type hinting
from collections.abc import Collection
# default dictionary for point hashing
from collections import defaultdict
# conversion from energy gradient units to m/s for velocity
velocity_units = 1e-3 * eV * angstrom / hbar


class BandStructure:
    """
    Contains bandstructure information for a given material.

    In addition to the dispersion relation and general parameters, this
    class also contains methods for discretizing the Fermi surface and
    calculating electronic properties.

    Parameters
    ----------
    dispersion : str
        The dispersion relation. Expresses the dispersion relation
        in terms of symbols in `wavevector_names` and additional
        parameters in `bandparams`. It must be parsable and
        differentiable by `sympy`. Energy units are milli eV.
    chemical_potential : float
        The chemical potential in milli eV.
    unit_cell : Collection[float]
        The dimensions of the unit cell in angstrom.
    atoms_per_cell : int, optional
        The number of atoms in the specified unit cell. This is not
        necessarily the exact number of atoms; it should be the number
        of conducting units in the cell. So, for example, this is equal
        to 2 for LSCO, which has the cuprate atoms in a BCC cell.
    bandparams : dict, optional
        The parameters of the dispersion relation. Energy units are
        milli eV and distance units are angstrom.
    axis_names : str or Collection[str], optional
        The names of the unit cell axes. Must be parsable by
        `sympy.symbols`.
    wavevector_names : str or Collection[str], optional
        The names of the wavevector components. Must be parsable by
        `sympy.symbols`.
    resolution : int, optional
        Controls the resolution of the grids used for discretizing the
        Fermi surface.

    Attributes
    ----------
    dispersion : str
        The dispersion relation. Updating this will automatically
        update `energy_func` and `velocity_func`.
    chemical_potential : float
        The chemical potential in milli eV.
    unit_cell : Collection[float]
        The dimensions of the unit cell in angstrom.
    atoms_per_cell : int
        The number of atoms (more precisely, the number of conducting
        units) in the specified unit cell.
    bandparams : dict
        The parameters of the dispersion relation. Updating this 
        will automatically update `energy_func` and `velocity_func`.
    energy_func : function
        The energy function for the dispersion relation. Takes
        kx, ky, and kz in angstrome^-1 as arguments and returns
        the energy in milli eV.
    velocity_func : function
        The velocity function for the dispersion relation. Takes
        kx, ky, and kz in angstrome^-1 as arguments and returns
        the velocity vector as a list [vx, vy, vz] in units of m/s.
    kpoints : (N, 3) numpy.ndarray
        The discretized k-points on the Fermi surface. Each row
        corresponds to a k-point in the form [kx, ky, kz].
    kfaces : (F, 3) numpy.ndarray
        The faces of the triangulated surface in k-space. Each row
        corresponds to a face in the form [i, j, k], where i, j,
        and k are the indices of the vertices of the face in the
        `kpoints` array.
    kpoints_periodic : (N, 3) numpy.ndarray of float
        The kpoints on the Fermi surface with the duplicate boundary
        points removed.
    kfaces_periodic : (F, 3) numpy.ndarray of int
        Same as `kfaces`, but points to the unique points in
        kpoints_periodic.
    resolution : int
        The resolution of the grids used for approximating the Fermi
        surface in the various steps.
    axis_names : str or Collection[str]
        The names of the unit cell axes.
    wavevector_names : str or Collection[str]
        The names of the wavevector components.
    """
    def __init__(
            self, dispersion: str, chemical_potential: float,
            unit_cell: Collection[float], atoms_per_cell: int = 1,
            bandparams: dict = {},
            axis_names: Collection[str] | str = ['a', 'b', 'c'],
            wavevector_names: Collection[str] | str = ['kx', 'ky', 'kz'],
            resolution: int = 20, **kwargs):
        # avoid triggering the __setattr__ method for the first time
        super().__setattr__('dispersion', dispersion)
        super().__setattr__('bandparams', bandparams)
        self.chemical_potential = chemical_potential
        self.unit_cell = unit_cell
        self.atoms_per_cell = atoms_per_cell
        self.axis_names = axis_names
        self.wavevector_names = wavevector_names
        self.resolution = resolution
        self._parse_dispersion()
        self.kpoints = np.empty((0, 3))

    def __setattr__(self, name, value):
        if name == 'dispersion' or name == 'bandparams':
            super().__setattr__(name, value)
            self._parse_dispersion()
        if name in ['chemical_potential', 'unit_cell', 'resolution']:
            self.kpoints = np.empty((0, 3))
        super().__setattr__(name, value)
    
    def discretize(self):
        """
        Discretize the Fermi surface.

        First, a point cloud is generated by detecting sign changes in
        the energy minus chemical potential function. Then, the point
        cloud is downsampled to make the spacing more uniform. Finally,
        a marching cubes algorithm augmented with RIMLS is used to
        generate the final triangulated surface. After the surface
        construction, periodic boundary conditions are applied to
        "stitch" the open ends of the surface together.
        """
        self._gvec = np.array([np.pi / a for a in self.unit_cell])
        self._voxel_size = 2 * self._gvec / np.array(self.resolution)

        self._generate_point_cloud()
        normals = np.column_stack(self.velocity_func(
            self.kpoints[:, 0], self.kpoints[:, 1], self.kpoints[:, 2]))
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=self.kpoints, v_normals_matrix=normals))
        ms.apply_filter("generate_marching_cubes_rimls",
                        resolution=self.resolution)
        ms.apply_filter("meshing_close_holes")
        self.kpoints = ms.current_mesh().vertex_matrix()
        self.kfaces = ms.current_mesh().face_matrix()
        self.kpoints = self.kpoints[np.unique(self.kfaces)]
        self._stitch_periodic_boundaries()
    
    def periodic_distance(self, k1: np.ndarray, k2: np.ndarray,
                          broadcast: bool = False) -> np.ndarray:
        """
        Calculate the periodic distance between two k-points.

        Parameters
        ----------
        k1 : np.ndarray
            The first k-point.
        k2 : np.ndarray
            The second k-point.
        broadcast : bool, optional
            If True, assumes that k1 and k2 are 2D arrays containing
            multiple k-points and broadcasts the calculation
            accordingly. If False, assumes that k1 and k2 are 1D
            arrays containing a single k-point.

        Returns
        -------
        float
            k2 - k1 with periodic boundary conditions applied.
        """
        gvec = self._gvec[None, :] if broadcast else self._gvec
        kdiff = k2 - k1
        kdiff += gvec
        kdiff %= 2 * gvec
        kdiff -= gvec
        return kdiff

    def calculate_mass(self):
        """
        Calculate the effective mass of the charge carries.

        Returns
        -------
        float
            The effective mass divided by the rest mass of
            the electron, m_e.
        """
        # Placeholder for actual calculation
        return 0.0
    
    def _parse_dispersion(self):
        """
        Parse the dispersion relation and extract the necessary
        information for further calculations.
        """
        ksymbols = sympy.symbols(self.wavevector_names)
        all_symbols = (ksymbols + sympy.symbols(self.axis_names)
                       + sympy.symbols(list(self.bandparams.keys())))
        self._energy_sympy = sympy.sympify(self.dispersion)
        self._velocities_sympy = [
            sympy.diff(self._energy_sympy, k) * velocity_units
            for k in sympy.symbols(self.wavevector_names)]
        for i, v in enumerate(self._velocities_sympy):
            if v == 0:
                self._velocities_sympy[i] = f"numpy.zeros_like({ksymbols[i]})"
        self._energy_func_full = sympy.lambdify(
            all_symbols, self._energy_sympy)
        self._velocity_funcs_full = [
            sympy.lambdify(all_symbols, vexpr, 'numpy')
            for vexpr in self._velocities_sympy]
        self.energy_func = lambda kx, ky, kz: self._energy_func_full(
            kx, ky, kz, *self.unit_cell, **self.bandparams)
        self.velocity_func = lambda kx, ky, kz: [
            vfunc(kx, ky, kz, *self.unit_cell, **self.bandparams)
            for vfunc in self._velocity_funcs_full]

    def _stitch_periodic_boundaries(self, threshold=1e-5):
        """
        Find duplicate points on the periodic boundaries, then make the
        periodic mesh arrays. Threshold sets the fraction of the
        resolution that we consider the points to be the same if they
        are within that distance.
        """
        voxel_coordinates = np.round((self.kpoints+self._gvec[None,:])
                                     / (threshold*self._voxel_size[None,:]))
        voxel_coordinates %= np.round(
            2*self._gvec / (threshold*self._voxel_size))[None, :]
        point_bins = defaultdict(list)
        duplicate_points = dict()
        for i, coordinates in enumerate(voxel_coordinates):
            point_bins[tuple(coordinates)].append(i)
        for coordinates in point_bins:
            point_bin = point_bins[coordinates]
            num_neighbors = len(point_bin)
            if num_neighbors > 1:
                primary_point = point_bin[0]
                for point in point_bin[1:]:
                    duplicate_points[point] = primary_point
        self._build_periodic_mesh(duplicate_points)
    
    def _build_periodic_mesh(self, duplicate_points):
        """
        Build the periodic kpoints and kfaces arrays by removing
        duplicate points and reindexing.
        """
        unique_mask = np.full(len(self.kpoints), True)
        unique_mask[list(duplicate_points.keys())] = False
        self.kpoints_periodic = self.kpoints[unique_mask]
        reindex_map = np.cumsum(unique_mask) - 1
        self.kfaces_periodic = np.empty_like(self.kfaces)
        for i, face in enumerate(self.kfaces):
            for j, point in enumerate(face):
                if point in duplicate_points:
                    reindex_map[point] = reindex_map[
                        duplicate_points[point]]
                self.kfaces_periodic[i, j] = reindex_map[point]

    def _generate_point_cloud(self):
        """
        Generate a point cloud of the Fermi surface by detecting sign
        changes and applying simple interpolation.
        """
        kgrid = np.mgrid[-self._gvec[0]:self._gvec[0]:1j*self.resolution,
                         -self._gvec[1]:self._gvec[1]:1j*self.resolution,
                         -self._gvec[2]:self._gvec[2]:1j*self.resolution]
        energy_diff = self.energy_func(*kgrid) - self.chemical_potential
        
        kpoints = []
        for axis in range(3):
            energy_diff_shifted = np.roll(energy_diff, -1, axis=axis)
            mask = (energy_diff * energy_diff_shifted < 0)

            indexer = [slice(None)] * 3
            indexer[axis] = slice(-1, None)
            mask[tuple(indexer)] = False

            k1 = [kgrid[i][mask] for i in range(3)]
            k2 = [np.roll(kgrid[i], -1, axis=axis)[mask] for i in range(3)]
            f1 = energy_diff[mask]
            f2 = energy_diff_shifted[mask]

            alpha = f1 / (f1 - f2)  # interpolation weight
            interpolated = [k1[i] + alpha * (k2[i] - k1[i]) for i in range(3)]

            kpoints.append(np.stack(interpolated, axis=-1))

        self.kpoints = np.concatenate(kpoints, axis=0)
        self._voxel_grid_downsample()

    def _voxel_grid_downsample(self):
        voxel_indices = np.floor(self.kpoints / self._voxel_size).astype(int)
        voxel_dict = defaultdict(list)
        for i, voxel in enumerate(voxel_indices):
            voxel_dict[tuple(voxel)].append(i)
        self.kpoints = np.array([self.kpoints[indices].mean(axis=0)
                                 for indices in voxel_dict.values()])
