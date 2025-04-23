import numpy as np
import sympy
# marching squares
from skimage.measure import find_contours
# units
from scipy.constants import hbar, eV, angstrom
# type hinting
from typing import Union
from collections.abc import Collection
# conversion from energy gradiennt units to m/s for velocity
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
    atoms_per_cell : int
        The number of atoms in the specified unit cell. This is not
        necessarily the exact number of atoms; it should be the number
        of conducting units in the cell. So, for example, this is equal
        to 2 for LSCO, which has the cuprate atoms in a BCC cell.
        The default is 1.
    bandparams : dict, optional
        The parameters of the dispersion relation. Energy units are
        milli eV and distance units are angstrom. The default is {}.
    axis_names : str or Collection[str], optional
        The names of the unit cell axes. Must be parsable by
        `sympy.symbols`. The default is ['a', 'b', 'c'].
    wavevector_names : str or Collection[str], optional
        The names of the wavevector components. Must be parsable by
        `sympy.symbols`. The default is ['kx', 'ky', 'kz'].
    res : int or collection of int, optional
        The "resolution" of the grid (side-length) used for
        approximating each 2D layer of the Fermi surface using the
        marching squares algorithm. If a single integer is given, it
        is taken as the resolution for both axes in k-space. If a
        collection of integers is given, each integer indicates the
        resolution for each axis in k-space. Can be set later when
        discretizing the Fermi surface. The default is 50.
    nlayers : int, optional
        The number of 2D layers for approximating the 3D Fermi
        surface. Can be set later when discretizing the Fermi
        surface. The default is 10.

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
    energy_func : callable
        The energy function for the dispersion relation. Takes
        kx, ky, and kz in angstrome^-1 as arguments and returns
        the energy in milli eV.
    velocity_func : callable
        The velocity function for the dispersion relation. Takes
        kx, ky, and kz in angstrome^-1 as arguments and returns
        the velocity vector as a list [vx, vy, vz] in units of m/s.
    res : Tuple[int, int]
        The resolution of the grid (side-length of each dimension) used
        for approximating each 2D layer of the Fermi surface using the
        marching squares algorithm. Updating this will automatically
        erase the current Fermi surface discretization.
    nlayers : int
        The number of 2D layers for approximating the 3D Fermi surface.
        updating this will automatically erase the current Fermi
        surface discretization.
    kx : list[numpy.ndarray]
        The x coordinates of the Fermi surface points for each layer.
    ky : list[numpy.ndarray]
        The y coordinates of the Fermi surface points for each layer.
    kz : numpy.ndarray
        The z coordinates of each layer of the Fermi surface.
    axis_names : str or Collection[str]
        The names of the unit cell axes.
    wavevector_names : str or Collection[str]
        The names of the wavevector components.
    """
    def __init__(
            self, dispersion: str, chemical_potential: float,
            unit_cell: Collection[float], atoms_per_cell: int = 1,
            bandparams: dict = {},
            axis_names: Union[Collection[str], str] = ['a', 'b', 'c'],
            wavevector_names: Union[Collection[str], str] = ['kx', 'ky', 'kz'],
            res: Union[int, Collection[int]] = 50, nlayers: int = 10,
            **kwargs):
        # avoid triggering the __setattr__ method for the first time
        super().__setattr__('dispersion', dispersion)
        super().__setattr__('bandparams', bandparams)
        self.chemical_potential = chemical_potential
        self.unit_cell = unit_cell
        self.atoms_per_cell = atoms_per_cell
        self.axis_names = axis_names
        self.wavevector_names = wavevector_names
        self.res = res
        self.nlayers = nlayers
        self._parse_dispersion()
        self.kx = []
        self.ky = []
        self.kz = np.empty(0)

    def __setattr__(self, name, value):
        if name == 'res':
            if isinstance(value, int):
                value = (value, value)
            elif isinstance(value, tuple) and len(value) == 2:
                pass
            else:
                raise ValueError("res must be an int or a tuple of two ints")
        elif name == 'dispersion' or name == 'bandparams':
            super().__setattr__(name, value)
            self._parse_dispersion()
        if name in ['chemical_potential', 'unit_cell', 'res', 'nlayers']:
            self.kx = []
            self.ky = []
            self.kz = np.empty(0)
        super().__setattr__(name, value)
    
    def discretize(self, res: Union[int, Collection[int], None] = None,
                   nlayers: Union[int, None] = None):
        """
        Discretize the Fermi surface based on the specified number
        of points aThe "resolution" of the grid (side-length) used for
            approximating each 2D layer of the Fermi surface using the
            marching squares algorithm.nd layers.

        Parameters
        ----------
        res : int, optional
            The "resolution" of the grid (side-length) used for
            approximating each 2D layer of the Fermi surface using the
            marching squares algorithm. If a single integer is given, it
            is taken as the resolution for both axes in k-space. If a
            collection of integers is given, each integer indicates the
            resolution for each axis in k-space. If not provided, takes
            the value from the class attribute.
        nlayers : int, optional
            The number of 2D layers for approximating the 3D Fermi surface.
            If not provided, takes the value from the class attribute.
        """
        self.res = res if res is not None else self.res
        self.nlayers = nlayers if nlayers is not None else self.nlayers

        # unit cell in reciprocal space, in angstrom^-1
        gx, gy, gz = [np.pi / a for a in self.unit_cell]
        self.kx = []
        self.ky = []
        self.kz = np.linspace(
            -self.atoms_per_cell * gz, self.atoms_per_cell * gz, self.nlayers)
        self._kgrid = np.ogrid[-gx:gx:self.res[1]*1j, -gy:gy:self.res[0]*1j]

        for layer in range(self.nlayers):
            self.kx.append(np.empty(0))
            self.ky.append(np.empty(0))
            # Find Fermi surface contours using marching squares
            contours = find_contours(self.energy_func(
                self._kgrid[0], self._kgrid[1], self.kz[layer]),
                self.chemical_potential)
            self._add_contours_in_order(contours, layer)
            
    
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
        all_symbols = (sympy.symbols(self.wavevector_names)
                       + sympy.symbols(self.axis_names)
                       + sympy.symbols(list(self.bandparams.keys())))
        self._energy_sympy = sympy.sympify(self.dispersion)
        self._velocities_sympy = [
            sympy.diff(self._energy_sympy, k) * velocity_units
            for k in sympy.symbols(self.wavevector_names)]
        self._energy_func_full = sympy.lambdify(
            all_symbols, self._energy_sympy)
        self._velocity_funcs_full = [sympy.lambdify(all_symbols, vexpr)
                                     for vexpr in self._velocities_sympy]
        self.energy_func = lambda kx, ky, kz: self._energy_func_full(
            kx, ky, kz, *self.unit_cell, **self.bandparams)
        self.velocity_func = lambda kx, ky, kz: [
            vfunc(kx, ky, kz, *self.unit_cell, **self.bandparams)
            for vfunc in self._velocity_funcs_full]
    
    def _add_contours_in_order(self, contours, layer):
        """Add contours to the kx and ky lists in order."""
        contour_idx = 0
        while contours:
            contour = contours.pop(contour_idx)
            self.kx[layer] = np.append(self.kx[layer], contour[:, 1])
            self.ky[layer] = np.append(self.ky[layer], contour[:, 0])
            # Find closest contour
            min_distance_squared = np.inf
            for (i, neighboring_contour) in enumerate(contours):
                delta_k = (contour[-1,:]-neighboring_contour[0,:]
                            ) % (2*np.pi)
                distance_squared = delta_k[0]**2 + delta_k[1]**2
                if distance_squared < min_distance_squared:
                    min_distance_squared = distance_squared
                    contour_idx = i
