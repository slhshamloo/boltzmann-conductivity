import re
from unittest import result
import numpy as np
import scipy
from typing import Mapping, Collection, Callable
from copy import copy
from scipy.special import sph_harm_y


class ScatteringKernel:
    """Base class for scattering kernels.
    
    This class should contain the following attributes and methods:

    Attributes
    ----------
    coeffs : np.ndarray
        The coefficients of the scattering kernel. The entry at (i, j)
        corresponds to the coefficient for the basis functions with
        indices i and j.
    
    Methods
    -------
    build_coeffs(params) -> np.ndarray
        Build the coefficients of the scattering kernel from the given
        parameters and set the ``coeffs`` attribute. Only necessary if
        building a kernel with explicit basis functions.
    eval_basis(index, kx, ky, kz) -> np.ndarray
        Evaluate the basis function with the given index at the given
        wavevector. Only necessary if you want to use explicit basis
        functions.
    decompose(band)
        Decompose the kernel into the basis functions and coefficients.
        Takes in a band object. Only necessary if there is no explicit
        basis functions. This sets ``eval_basis`` and whatever
        is needed to build the transformation from the reduced
        basis to the full FEM basis.
    """
    def __init__(self, params: Mapping):
        self.coeffs = self.build_coeffs(params)
    
    def build_coeffs(self, params: Mapping) -> np.ndarray:
        """Build the coefficients of the scattering kernel
        from the given parameters and set the ``coeffs`` attribute.

        Parameters
        ----------
        params : dict
            A dictionary of parameters needed to construct the
            scattering kernel.

        Returns
        -------
        np.ndarray
            The coefficients of the scattering kernel.
        """
        self.coeffs = params['coeffs']
        return self.coeffs
    
    def eval_basis(self, index, kx, ky, kz) -> np.ndarray:
        """Evaluate the basis function with the given index
        at the given wavevector.

        Parameters
        ----------
        index : int
            The index of the basis function to evaluate.
        kx, ky, kz : float
            The components of the wavevector with units of 1/angstrom.

        Returns
        -------
        np.ndarray
            The values of the basis functions at the given wavevector.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class IsotropicKernel(ScatteringKernel):
    """Scattering kernel that is just a constant function of the wavevectors
    of the incoming and outgoing states. This corresponds to isotropic
    scattering.

    Parameters
    ----------
    C : float
        The value of the kernel function at all wavevectors. Should be in
        units of angstrom^2 THz.
    """
    def __init__(self, C):
        self.C = C
        self.coeffs = np.array([[C]])

    def eval_basis(self, index, kx, ky, kz):
        return np.ones_like(kx)


class SphericalKernel(ScatteringKernel):
    """Scattering kernel based on real-valued spherical harmonics
    :math:`\\sqrt{2} \\Re Y^{|m|}_l(\\theta, \\phi)` and
    :math:`\\sqrt{2} \\Im Y^{|m|}_l(\\theta, \\phi)`. The first,
    cosine-like, basis function corresponds to positive m, while the
    second, sine-like, basis function corresponds to negative m.

    Parameters
    ----------
    params : dict
        A dictionary mapping tuples of tuples of integers,
        ``((l, m), (l', m'))``, to the corresponding coefficients of
        the scattering kernel. Note that since the scattering kernel
        should be Hermitian (real-symmetric in this case), you must
        only specify one coefficient for each pair of indices. The
        kernel will automatically fill in the other coefficient.
    """
    def __init__(self, params: Mapping):
        super().__init__(params)

    def build_coeffs(self, params: Mapping):
        matrix_dict = {}
        for (l, m), (l_prime, m_prime) in params.keys():
            matrix_dict[(l*(l+1) + m, l_prime*(l_prime+1) + m_prime)] = \
                params[((l, m), (l_prime, m_prime))]
        self._inv_idx_map = np.sort(np.unique(np.array(
            [[key[0], key[1]] for key in matrix_dict.keys()]).flatten()))
        index_map = {idx: i for i, idx in enumerate(self._inv_idx_map)}
        self.coeffs = np.zeros((len(index_map), len(index_map)))
        for (i, j), value in matrix_dict.items():
            self.coeffs[index_map[i], index_map[j]] = value
            self.coeffs[index_map[j], index_map[i]] = value
        return self.coeffs
    
    def eval_basis(self, index, kx, ky, kz):
        index = self._inv_idx_map[index]
        theta = np.arccos(kz / np.sqrt(kx**2 + ky**2 + kz**2))
        phi = np.arctan2(ky, kx)
        l = index // (index+1)
        m = index % (index+1)
        if m >= 0:
            return np.sqrt(2) * sph_harm_y(l, m, theta, phi).real
        else:
            return np.sqrt(2) * sph_harm_y(l, -m, theta, phi).imag


class CylindricalKernel(ScatteringKernel):
    """Scattering kernel based on real cylindrical harmonics
    :math:`\\cos(m\\phi)` and :math:`\\sin(m\\phi)`.

    Parameters
    ----------
    params : dict or np.ndarray
        A dictionary mapping ``(m, m')`` to the non-zero coefficients
        of the scattering kernel. For the cosine basis functions,
        ``m`` is non-negative, while for the sine basis functions,
        ``m`` is negative. Note that since the scattering kernel
        should be Hermitian, (real-symmetric in this case), you must
        only specify one coefficient for each pair of indices. The
        kernel will automatically fill in the other coefficient.
    """
    def __init__(self, params: Mapping):
        super().__init__(params)

    def build_coeffs(self, params: Mapping):
        matrix_dict = {}
        for (m, m_prime), value in params.items():
            if m < 0:
                idx_m = -2*m - 1
            else:
                idx_m = 2*m
            if m_prime < 0:
                idx_m_prime = -2*m_prime - 1
            else:
                idx_m_prime = 2*m_prime
            matrix_dict[(idx_m, idx_m_prime)] = value
        self._inv_idx_map = np.sort(np.unique(np.array(
            [[key[0], key[1]] for key in matrix_dict.keys()]).flatten()))
        index_map = {idx: i for i, idx in enumerate(self._inv_idx_map)}
        self.coeffs = np.zeros((len(index_map), len(index_map)))
        for (i, j), value in matrix_dict.items():
            self.coeffs[index_map[i], index_map[j]] = value
            self.coeffs[index_map[j], index_map[i]] = value
        return self.coeffs

    def eval_basis(self, index, kx, ky, kz):
        index = self._inv_idx_map[index]
        phi = np.arctan2(ky, kx)
        if index % 2 == 0:
            return np.cos(index//2 * phi)
        else:
            return np.sin(((index//2) + 1) * phi)


class LegendreKernel(ScatteringKernel):
    """Scattering kernel based on Legendre polynomials
    :math:`P_l(\\cos \\theta)`.

    To preserve normalization, this kernel actually uses the spherical
    harmonics :math:`Y^{m=0}_l(\\theta, \\phi)`, which are proportional
    to the Legendre polynomials :math:`P_l(\\cos \\theta)`.

    Parameters
    ----------
    params : dict
        Either a dictionary mapping ``(l, l')`` to the non-zero
        coefficients of the scattering kernel, or a 2D array of
        coefficients where the entry at (l, l') corresponds to the
        coefficient for the basis functions with indices ``l`` and
        ``l'``. Note that since the scattering kernel should be
        Hermitian (real-symmetric in this case), if you use a
        dictionary, you must only specify one coefficient for each
        pair of indices. The kernel will automatically fill in the
        other coefficient.
    """
    def __init__(self, params: Mapping):
        super().__init__(params)

    def build_coeffs(self, params: Mapping):
        if isinstance(params, dict):
            matrix_dict = {}
            for (l, l_prime), value in params.items():
                matrix_dict[(l, l_prime)] = value
            self._inv_idx_map = np.sort(np.unique(np.array(
                [[key[0], key[1]] for key in matrix_dict.keys()]).flatten()))
            index_map = {idx: i for i, idx in enumerate(self._inv_idx_map)}
            self.coeffs = np.zeros((len(index_map), len(index_map)))
            for (i, j), value in matrix_dict.items():
                self.coeffs[index_map[i], index_map[j]] = value
        else:
            self.coeffs = params['coeffs']
        self.coeffs += self.coeffs.conj().T - np.diag(self.coeffs.diagonal())
        return self.coeffs

    def eval_basis(self, index, kx, ky, kz):
        index = self._inv_idx_map[index]
        theta = np.arccos(kz / np.sqrt(kx**2 + ky**2 + kz**2))
        return np.real(sph_harm_y(index, 0, theta, 0))


class AzimuthalHotspotKernel(ScatteringKernel):
    """Scattering kernel based on a Gaussian of the difference of the
    angles of the wavevectors of the incoming and outgoing states in
    the x-y plane with "hotspot" pairs.

    .. math::
    C(\\phi, \\phi') = \sum_i C_{hi}\sum_j \\mathrm{exp}\\left(
        -\\frac{(\phi-\phi_{hj})^2}{2\\sigma_h^2}\\right)
        \\mathrm{exp}\\left(-\\frac{(\phi-\phi'_{hj})^2}
        {2\\sigma_h^2}\\right)
    
    Pairs of "hotspots" separated by a particular angle will contribute
    to the scattering kernel in the form of a double Gaussian peak.

    Parameters
    ----------
    phi_h: Sequence of float
        The position of the hotspots by their angle in the x-y plane.
        In degrees, but converted to radians internally.
    dphi_h : float
        The connecting angle between the hotspots. In degrees, but
        converted to radians internally. Only pairs of hotspots with
        this particular connecting angle will contribute to the
        scattering kernel.
    C_h : float
        The amplitude of each Gaussian pair in the scattering kernel.
        In units of angstrom^2 THz.
    sigma_h : Sequence of float
        The width of the Gaussians (in radians).
    tol : float
        The tolerance for determining whether a pair of hotspots is
        connected by the given connecting angle. This is in absolute
        terms, in radians.
    """
    def __init__(self, phi_h, dphi_h, C_h, sigma_h, tol=1e-5):
        self.phi_h = np.radians(np.array(phi_h))
        self.dphi_h = np.radians(dphi_h)
        self.C_h = C_h
        self.sigma_h = sigma_h
        self.tol = tol
        self.build_coeffs()

    def build_coeffs(self):
        self.coeffs = np.zeros((len(self.phi_h), len(self.phi_h)))
        for i in range(len(self.phi_h)):
            for j in range(i, len(self.phi_h)):
                phi_diff = _make_angle_periodic(self.phi_h[i] - self.phi_h[j])
                if abs(abs(phi_diff) - self.dphi_h) < self.tol:
                    self.coeffs[i, j] = self.C_h
                    self.coeffs[j, i] = self.C_h

    def eval_basis(self, index, kx, ky, kz):
        phi = np.arctan2(ky, kx)
        hotspot = _make_angle_periodic(self.phi_h[index])
        phi_diff = _make_angle_periodic(phi - hotspot)
        return np.exp(-phi_diff**2 / (2*self.sigma_h**2))


class CustomKernel(ScatteringKernel):
    """Scattering kernel expanded by the eigenvalue decomposition of
    a custom kernel function.

    Parameters
    ----------
    kernel_func:
        Takes in wavevector components (kx, ky, kz, kx', ky', kz')
        and returns the value of the kernel function at that wavevector.
        The output should be in units of angstrom^2 THz. The wavevector
        components have units of 1/angstrom.
    rank:
        The number of eigenvalue and eigenvector pairs
        to keep when decomposing the kernel function into basis
        functions and coefficients. This would be the final rank of
        the resulting scattering kernel.
    ``'low_res'``:
        The resolution of the approximate band object
        used for a more managable eigenvalue decomposition.
    """
    def __init__(self, kernel_func: Callable, rank: int = 20,
                 low_res: int = 21, **kwargs):
        self.kernel_func = kernel_func
        self.rank = rank
        self.low_res = low_res
        self.coeffs = None
    
    def decompose(self, band):
        band_decomp = copy(band)
        band_decomp.resolution = self.low_res
        band_decomp.discretize()
        low_to_high = _interpolation_matrix(band.kpoints, band_decomp.kpoints)

        kx, ky, kz = band_decomp.kpoints.T
        kx, ky, kz = kx[:, None], ky[:, None], kz[:, None]
        kx_prime, ky_prime, kz_prime = kx.T, ky.T, kz.T
        kernel_matrix = self.kernel_func(
            kx, ky, kz, kx_prime, ky_prime, kz_prime)

        matvec = scipy.sparse.linalg.LinearOperator(
            kernel_matrix.shape, matvec=lambda x: kernel_matrix @ x)
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            matvec, k=self.rank)
        
        self.coeffs = np.diag(eigenvalues)
        self.eigenvectors = eigenvectors
        self.projector = low_to_high @ eigenvectors
        return self.coeffs


class SumKernel(ScatteringKernel):
    """Scattering kernel that is a sum of other kernels.
    
    The resulting basis is a direct sum of each basis for the individual
    kernels. This means that the vector of the basis functions is just a
    concatenation of the basis functions for each kernel, and the
    coefficient matrix would become a block-diagonal matrix with the
    coefficient matrices of the individual kernels as blocks.

    Custom kernels will just be summed together in one unified function,
    and the basis obtained from the decomposition of the resulting
    kernel will be used.

    Parameters
    ----------
    kernels : list of ScatteringKernel
        The kernels to sum together.
    decomp_params : dict
        The parameters for the decomposition of the custom kernels.
    """
    def __init__(self, kernels: Collection[ScatteringKernel],
                 **decomp_params):
        self.custom_kernel = None
        custom_kernels = [k for k in kernels if isinstance(k, CustomKernel)]
        if len(custom_kernels) > 0:
            self.custom_kernel_sum = CustomKernelSumCallable(
                [k.kernel_func for k in custom_kernels])
            self.custom_kernel = CustomKernel(
                self.custom_kernel_sum, **decomp_params)
        
        self.explicit_kernels = [
            k for k in kernels if not isinstance(k, CustomKernel)]
        if len(self.explicit_kernels) > 0:
            self.coeffs = scipy.linalg.block_diag(
                *[k.coeffs for k in self.explicit_kernels])
    
    def eval_basis(self, index, kx, ky, kz):
        current_index = 0
        for kernel in self.explicit_kernels:
            size = kernel.coeffs.shape[0]
            if index < current_index + size:
                return kernel.eval_basis(index - current_index, kx, ky, kz)
            current_index += size
        raise IndexError("Index out of range for the combined basis.")

    def decompose(self, band):
        if self.custom_kernel is not None:
            self.custom_kernel.decompose(band)


class CustomKernelSumCallable:
    """
    Helper class to sum multiple custom kernel functions
    together into one unified function.
    """
    def __init__(self, kernels: Collection[Callable]):
        self.kernels = kernels
    def __call__(self, kx, ky, kz, kx_prime, ky_prime, kz_prime):
        result = 0
        for kernel in self.kernels:
            result += kernel(kx, ky, kz, kx_prime, ky_prime, kz_prime)
        return result


class AzimuthalKernelFunction:
    """Scattering kernel that is a function of the angle of the
    wavevector in the x-y plane, but not the magnitude.

    .. math::
    C(\\phi, \\phi') = C_1\\left|\\mathrm{cos}\\left(m
        \\left[\\frac{\\phi+\\phi'}{2}-\\phi_c\\right]\\right)
        \\right|^{\\nu_c}
    
    Call the object like ``C(kx, ky, kz, kx_prime, ky_prime, kz_prime)``
    to evaluate the kernel function at the given wavevectors.

    Parameters
    ----------
    C_1 : float
        The coefficient for the anisotropic term in the kernel function.
    m : int
        Sets the symmetry of the anisotropy over the angle in the x-y
        plane. For example, ``m=2`` repeats the peak every 90 degrees.
    nu : float
        Sets the sharpness of the anisotropy in the kernel function.
    phi_0 : float
        The phase shift of the anisotropy in the kernel function.
    """
    def __init__(self, C_1, m=1, nu=1.0, phi_0=0.0):
        self.C_1 = C_1
        self.m = m
        self.nu = nu
        self.phi_0_rad = np.radians(phi_0)

    def __call__(self, kx, ky, kz, kx_prime, ky_prime, kz_prime):
        phi = np.arctan2(ky, kx)
        phi_prime = np.arctan2(ky_prime, kx_prime)
        phi_mean = (phi+phi_prime) / 2
        # keep the angle between -pi and pi
        phi_mean = _make_angle_periodic(phi_mean)
        return self.C_1 * np.abs(
            np.cos(self.m*phi_mean - self.phi_0_rad))**self.nu


class GaussianScattering:
    """Scattering kernel based on a Gaussian of the difference of the
    wavevectors of the incoming and outgoing states.

    .. math::
        C(k, k') = C \\mathrm{exp}(
        -(|\\mathbf{k}\\pm \\mathbf{k'}|-delta)^2/2\\sigma^2)

    Call the object like ``C(kx, ky, kz, kx_prime, ky_prime, kz_prime)``
    to evaluate the kernel function at the given wavevectors.

    Parameters
    ----------
    C : float
        The amplitude of the Gaussian.
    sigma : float
        The width of the Gaussian (in angstroms).
    backward : bool
        Whether the Gaussian is a function of |k+k'| (backward scattering)
        or |k-k'| (forward scattering). Default is False, which corresponds
        to forward scattering.
    delta : float
        The shift of the Gaussian from zero. This can be used to model
        scattering that is peaked at a non-zero momentum transfer, such
        as forward scattering (delta = 0) or backward scattering (delta
        = 2|k|).
    """
    def __init__(self, C, sigma, delta=0.0, backward=False):
        self.C = C
        self.sigma = sigma
        self.delta = delta
        self.backward = backward

    def __call__(self, kx, ky, kz, kx_prime, ky_prime, kz_prime):
        sign = -1 if self.backward else 1
        diff_x = kx - sign * kx_prime
        diff_y = ky - sign * ky_prime
        diff_z = kz - sign * kz_prime
        diff_abs = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

        return self.C * np.exp(-(diff_abs-self.delta)**2 / (2*self.sigma**2))


class AnisotropicGaussianScattering:
    """Scattering kernel based on a Gaussian of the difference of the
    angles of the wavevectors of the incoming and outgoing states in
    the x-y plane, with anisotropic parameters.

    .. math::
    C(\\phi, \\phi') = \\left(C_0 + C_1 \\left|\\mathrm{cos}\\left(m
        \\left[\\frac{\\phi+\\phi'}{2}-\\phi_c\\right]\\right)
        \\right|^{\\nu_c}\\right)\\mathrm{exp}\\left(
        -\\frac{(|\\phi-\\phi'|-\\delta)^2}{2\\left(
        \\sigma_0+\\sigma_1\\left|\\mathrm{cos}\\left(
        m\\left[\\frac{\\phi+\\phi'}{2}-\\phi_s\\right]\\right)
        \\right|^{\\nu_s}\\right)^2}\\right)

    Call the object like ``C(kx, ky, kz, kx_prime, ky_prime, kz_prime)``
    to evaluate the kernel function at the given wavevectors.

    Parameters
    ----------
    C_0 : float
        The constant term in the amplitude of the Gaussian.
    C_1 : float
        The coefficient for the anisotropic term in the amplitude of the
        Gaussian.
    sigma_0 : float
        The constant term in the width of the Gaussian (in radians).
    sigma_1 : float
        The coefficient for the anisotropic term in the width of the
        Gaussian (in radians).
    m : int
        Sets the symmetry of the anisotropy over the angle in the x-y
        plane. For example, ``m=2`` repeats the peak every 90 degrees.
    nu_c : float
        Sets the sharpness of the anisotropy in the amplitude of
        the Gaussian.
    nu_s : float
        Sets the sharpness of the anisotropy in the width of the
        Gaussian.
    phi_c : float
        Sets the angle at which the peak of the amplitude of the
        anisotropy occurs.  In units of degrees.
    phi_s : float
        Sets the angle at which the width of the kernel is largest.
        In units of degrees.
    delta : float
        The shift of the Gaussian from zero, in degrees. This can be
        used to model scattering that is peaked at a non-zero angle
        difference, such as forward scattering (``delta=0``) or
        backward scattering (``delta=180``).
    """
    def __init__(self, C_0, C_1, sigma_0, sigma_1, m=1,
                 nu_c=1.0, nu_s=1.0, phi_c=0.0, phi_s=0.0, delta=0.0):
        self.C_0 = C_0
        self.C_1 = C_1
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.m = m
        self.nu_c = nu_c
        self.nu_s = nu_s
        self.phi_c_rad = np.radians(phi_c)
        self.phi_s_rad = np.radians(phi_s)
        self.delta_rad = np.radians(delta)
    def __call__(self, kx, ky, kz, kx_prime, ky_prime, kz_prime):
        phi = np.arctan2(ky, kx)
        phi_prime = np.arctan2(ky_prime, kx_prime)
        phi_mean = (phi+phi_prime) / 2
        phi_diff = phi - phi_prime
        # keep the angles between -pi and pi
        phi_mean = _make_angle_periodic(phi_mean)
        phi_diff = _make_angle_periodic(phi_diff)

        amplitude = self.C_0 + self.C_1 * np.abs(np.cos(
            self.m * (phi_mean-self.phi_c_rad))) ** self.nu_c
        width = self.sigma_0 + self.sigma_1 * np.abs(np.cos(
            self.m * (phi_mean-self.phi_s_rad))) ** self.nu_s
        phi_diff = abs(phi_diff) - self.delta_rad
        return amplitude * np.exp(-phi_diff**2 / (2 * width**2))


def build_kernel(kernel, kernel_params):
    """Build a scattering kernel based on the given kernel type and
    parameters.

    Parameters
    ----------
    kernel : str or list of str
        The type(s) of kernel(s) to build. Kernels with explicit basis
        functions include:

        * | ``'spherical'``: Spherical Harmonics. The parameters are
          | indicated by a pair of tuples of integers,
          | ``((l, m), (l', m'))``, mapping to the corresponding
          | coefficients of the (real-valued) spherical harmonics
          | :math:`P_l^m(\\theta, \\phi)`
          | and :math:`P_{l'}^{m'}(\\theta, \\phi)`.
        * | ``'cylindrical'``: Cylindrical Harmonics, which are just
          | cosines and sines of the angle in the x-y plane. The
          | parameter dictionary keys can be expressed in two ways;
          | first, as a pair of integers ``(m, m')``, where
          | non-negative m corresponds to cosines and negative m
          | corresponds to sines; second, as a string of the form
          | ``'[cos/sin][m][cos/sin][m']``, so for example (-2, 3)
          | in the previous scheme would correspond to ``'sin2cos3'``.
          | You can set single sines and cosines either by setting the
          | other ``m`` to zero or by simply omitting it, so for
          | example ``'cos3'``. You can set the constant term by having
          | both m and m' be zero, or just use ``'1'`` or ``'constant'``
          | or ``'const'`` or ``'iso'`` as the key.
        * | ``'legendre'``: Legendre polynomials. The parameters are
          | indicated by a pair of integers, ``(l, l')``, mapping to
          | the corresponding coefficients of the Legendre polynomials
          | :math:`P_l(\\cos \\theta)`
          | and :math:`P_{l'}(\\cos \\theta)`. Note that, to keep the
          | normalization consistent, this kernel actually uses the
          | spherical harmonics :math:`Y^{m=0}_l(\\theta, \\phi)`.
        
        Kernels without explicit basis functions include:

        * | ``'isotropic'``: Isotropic scattering, where the kernel is
          | just a constant value ``'C_0'`` at all wavevectors.
        * | ``'azimuthal'``: A kernel that is a function of the angle
          | of the wavevector in the x-y plane, but not the magnitude.
          | The function has the form
          | :math:`C(\\phi, \\phi') = C_1|\\mathrm{cos}(m[\\frac{\\phi+\\phi'}{2}-\\phi])|^\\nu`.
          | The kernel parameters are ``'C_1'`` (the aplitude),
          | ``'m'`` (the symmetry), ``'nu'`` (the sharpness of the
          | anisotropy), and ``'phi'`` (the angle at which the
          | anisotropy peaks, in degrees).
        * | ``'forward'``: Forward scattering expressed as a Gaussian
          | :math:`C_f \\mathrm{exp}(-|k-k'|^2/2\\sigma_f^2)`. The
          | kernel parameters are ``'C_f'``, which sets the amplitude,
          | and ``'sigma_f'``, which sets the width.
        * | ``'backward'``: Backward scattering expressed as a Gaussian
          | :math:`C_b \\mathrm{exp}(-|k+k'|^2/2\\sigma_b^2)`. The
          | kernel parameters are ``'C_b'``, which sets the amplitude,
          | and ``'sigma_b'``, which sets the width.
        * | ``'forward_phi'``: Like ``'forward'``, but the Gaussian is
          | in the angle of the wavevector in the x-y plane instead of
          | the full wavevector. So,
          | :math:`C_f \\mathrm{exp}(-|\\phi-\\phi'|^2/(2\\sigma_f^2))`.
        * | ``'backward_phi'``: Like ``'backward'``, but the Gaussian
          | is in the angle of the wavevector in the x-y plane instead
          | of the full wavevector. So,
          | :math:`C_b \\mathrm{exp}(-|\\phi+\\phi'|^2/(2\\sigma_b^2))`.
        * | ``'forward_anisotropic'``: Like ``'forward_phi'``, but with
          | anisotropic parameters for the Gaussian. This means,
          | :math:`C_f = C_{f0} + C_{f1}|\\mathrm{cos}(m[(\\phi+\\phi')/2-\\phi_{fc}])|^{\\nu_{fc}}` and
          | :math:`\\sigma_f=\\sigma_{f0}+\\sigma_{f1}\\left|\\mathrm{cos}(m[(\\phi+\\phi')/2-\\phi_{fs}])\\right|^{\\nu_{fs}}`.
          | The kernel parameters are ``'C_f0'``, ``'C_f1'``,
          | ``'sigma_f0'``, ``'sigma_f1'``, ``'m'``, ``\\nu_{fc}``,
          | ``\\nu_{fs}``, ``'phi_fc'``, and ``'phi_fs'``.
          | Any omitted parameters will be set to zero, except for ``'m'``,
          | which will be set to 1 by default.
        * | ``'backward_anisotropic'``: Like ``'backward_phi'``, but
          | with anisotropic parameters for the Gaussian. This means,
          | :math:`C_b = C_{b0} + C_{b1}|\\mathrm{cos}(m[(\\phi+\\phi')/2-\\phi_{bc}])|^{\\nu_{bc}}` and
          | :math:`\\sigma_b=\\sigma_{b0}+\\sigma_{b1}\\left|\\mathrm{cos}(m[(\\phi+\\phi')/2-\\phi_{bs}])\\right|^{\\nu_{bs}}`.
          | The kernel parameters are ``'C_b0'``, ``'C_b1'``,
          | ``'sigma_b0'``, ``'sigma_b1'``, ``'m'``, ``\\nu_{bc}``,
          | ``\\nu_{bs}``, ``'phi_bc'``, and ``'phi_bs'``.
          | Any omitted parameters will be set to zero, except for ``'m'``,
          | which will be set to 1 by default.
        * | ``'hotspot_phi'``: See
          | `elecboltz.kernel.AzimuthalHotspotScattering` for details.

        If a list of kernels is provided, the resulting kernel is the
        sum of all the kernels in the list.

    kernel_params : dict or list of dict
        The parameters for the kernel(s). If a list of explicit basis
        kernels is provided, a list of parameter dictionaries should
        be provided, where each dictionary corresponds to the
        parameters for the respective kernel. For non-explicit kernels,
        an extra dictionary can be provided at the end of the list to
        specify the parameters for the decomposition of the resulting
        kernel (see `elecboltz.kernel.CustomKernel`).
    """
    if isinstance(kernel, list):
        decomp_params = {}
        if len(kernel_params) == len(kernel) + 1:
            decomp_params = kernel_params[-1]
            kernel_params = kernel_params[:-1]
        elif len(kernel_params) != len(kernel):
            raise ValueError(
                "If a list of kernels is provided, the number of parameter "
                "dictionaries should either be the same as the number of "
                "kernels or one more than the number of kernels (if the last "
                "dictionary is for the decomposition of the "
                "resulting kernel).")
        kernels = []
        for k, p in zip(kernel, kernel_params):
            kernels.append(build_kernel(k, p))
        return SumKernel(kernels, **decomp_params)
    kernel_builders = [_build_explicit_kernel, _build_anisotropic_gaussian_kernel,
                       _build_gaussian_kernel]
    for builder in kernel_builders:
        built_kernel = builder(kernel, kernel_params)
        if built_kernel is not None:
            return built_kernel
    raise ValueError(f"Unsupported kernel type: {kernel}")


def _build_explicit_kernel(kernel, kernel_params):
    if kernel == 'isotropic':
        return IsotropicKernel(kernel_params['C_0'])
    elif kernel == 'hotspot_phi':
        return AzimuthalHotspotKernel(
            kernel_params['phi_h'], kernel_params['dphi_h'],
            kernel_params['C_h'], kernel_params['sigma_h'],
            kernel_params.get('tol', 1e-5))
    elif kernel == 'spherical':
        return SphericalKernel(kernel_params)
    elif kernel == 'cylindrical':
        if isinstance(list(kernel_params.keys())[0], str):
            new_params = {}
            for key, value in kernel_params.items():
                new_params[_get_cylindrical_indices(key)] = value  
            kernel_params = new_params
        return CylindricalKernel(kernel_params)
    elif kernel == 'legendre':
        return LegendreKernel(kernel_params)
    return None


def _build_gaussian_kernel(kernel, kernel_params):
    if kernel == 'azimuthal':
        return CustomKernel(AzimuthalKernelFunction(
            kernel_params['C_1'], kernel_params.get('m', 1),
            kernel_params.get('nu', 1), kernel_params.get('phi', 0)),
            **kernel_params)
    elif kernel == 'forward':
        return CustomKernel(GaussianScattering(
            kernel_params['C_f'], kernel_params['sigma_f']), **kernel_params)
    elif kernel == 'backward':
        return CustomKernel(GaussianScattering(
            kernel_params['C_b'], kernel_params['sigma_b'], backward=True),
            **kernel_params)
    elif kernel == 'forward_phi':
        return CustomKernel(AnisotropicGaussianScattering(
            kernel_params['C_f'], 0, kernel_params['sigma_f'], 0, 1, 0),
            **kernel_params)
    elif kernel == 'backward_phi':
        return CustomKernel(AnisotropicGaussianScattering(
            kernel_params['C_b'], 0, kernel_params['sigma_b'], 0, 1, 0,
            delta=180), **kernel_params)
    return None


def _build_anisotropic_gaussian_kernel(kernel, kernel_params):
    if kernel == 'forward_anisotropic':
        return CustomKernel(AnisotropicGaussianScattering(
            C_0=kernel_params.get('C_f0', 0), C_1=kernel_params.get('C_f1', 0),
            sigma_0=kernel_params['sigma_f0'],
            sigma_1=kernel_params.get('sigma_f1', 0),
            m=kernel_params.get('m', 1), nu_c=kernel_params.get('nu_fc', 1),
            nu_s=kernel_params.get('nu_fs', 1),
            phi_c=kernel_params.get('phi_fc', 0),
            phi_s=kernel_params.get('phi_fs', 0)),
            **kernel_params)
    elif kernel == 'backward_anisotropic':
        return CustomKernel(AnisotropicGaussianScattering(
            C_0=kernel_params.get('C_b0', 0), C_1=kernel_params.get('C_b1', 0),
            sigma_0=kernel_params['sigma_b0'],
            sigma_1=kernel_params.get('sigma_b1', 0),
            m=kernel_params.get('m', 1), nu_c=kernel_params.get('nu_bc', 1),
            nu_s=kernel_params.get('nu_bs', 1),
            phi_c=kernel_params.get('phi_bc', 0),
            phi_s=kernel_params.get('phi_bs', 0),
            delta=180), **kernel_params)
    return None


def _make_angle_periodic(angle):
    return (np.fmod(angle, np.pi) - np.sign(angle)
            * np.pi * np.fmod(np.abs(angle)//np.pi, 2))


def _get_cylindrical_indices(key):
    if key in ['1', 'constant', 'const', 'iso']:
        return (0, 0)
    else:
        match = re.match(r'(cos|sin)+(\d+)(cos|sin)?(\d*)', key)
        if not match:
            raise ValueError(
                f"Invalid cylindrical kernel parameter key: {key}")
        m1 = int(match.group(2)) if match.group(2) else 0
        m2 = int(match.group(4)) if match.group(4) else 0
        if match.group(1) == 'sin':
            m1 = -m1
        if match.group(3) == 'sin':
            m2 = -m2
        return (m1, m2)


def _interpolation_matrix(points, mesh, nearest_neighbors=8):
    tree = scipy.spatial.cKDTree(mesh)
    _, indices = tree.query(points, k=nearest_neighbors) # (N, K)
    nnpoints = mesh[indices] # (N, K, 3)
    nnpoints = nnpoints.swapaxes(-2, -1) # (N, 3, K)
    sol = scipy.linalg.lstsq(nnpoints, points[..., None])[0] # (N, K)
    sol /= sol.sum(axis=-1, keepdims=True) * nearest_neighbors # Normalization

    rows = np.repeat(np.arange(points.shape[0]), nearest_neighbors)
    cols = indices.reshape(-1)
    data = sol.reshape(-1)
    return scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(points), len(mesh)))
