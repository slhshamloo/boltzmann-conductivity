import re
import numpy as np
import scipy
from typing import Mapping, Collection, Callable
from copy import copy
if scipy.__version__ >= "1.15.0":
    from scipy.special import sph_harm_y
else:
    from scipy.special import sph_harm as sph_harm_y


class ScatteringKernel:
    """Base class for scattering kernels.
    
    This class should contain the following attributes and methods:

    Attributes
    ----------
    is_explicit : bool
        Whether the kernel already has explicit basis functions that
        can be evaluated at any wavevector. If ``False``, the kernel
        should provide a method ``decompose(band)`` to decompose the
        kernel into the basis functions and coefficients (effectively
        building ``ceoffs`` and ``eval_basis`` from the kernel
        function).
    coeffs : np.ndarray
        The coefficients of the scattering kernel. The entry at (i, j)
        corresponds to the coefficient for the basis functions with
        indices i and j.
    
    Methods
    -------
    build_coeffs(params) -> np.ndarray
        Build the coefficients of the scattering kernel from the given
        parameters and set the ``coeffs`` attribute. Only necessary if
        ``is_explicit`` is ``True``.
    eval_basis(index, kx, ky, kz) -> np.ndarray
        Evaluate the basis function with the given index at the given
        wavevector. Only necessary if ``is_explicit`` is ``True``.
        The output should be in units of angstrom^2 THz.
        The wavevector components have units of 1/angstrom.
    decompose(band)
        Decompose the kernel into the basis functions and coefficients.
        Takes in a band object. Only necessary if ``is_explicit`` is
        ``False``. This should set the ``coeffs`` attribute and provide
        the object with the parameters needed for ``eval_basis``.
    """
    def __init__(self, params: Mapping):
        self.is_explicit = True
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


class SumKernel(ScatteringKernel):
    """Scattering kernel that is a sum of other kernels.
    
    The resulting basis is a direct sum of each basis for the individual
    kernels. This means that the vector of the basis functions is just a
    concatenation of the basis functions for each kernel, and the
    coefficient matrix would become a block-diagonal matrix with the
    coefficient matrices of the individual kernels as blocks.
    
    Keep in mind that this creates many unused entries in the
    scattering matrix. So, always try finding a more general basis
    before simply adding multiple kernels with this method.

    Parameters
    ----------
    kernels : list of ScatteringKernel
        The kernels to sum together.
    """
    def __init__(self, kernels: Collection[ScatteringKernel]):
        self.is_explicit = True
        self.kernels = kernels
        self.coeffs = self.build_coeffs(kernels)
    
    def build_coeffs(self, kernels: Collection[ScatteringKernel]):
        total_size = sum(kernel.coeffs.shape[0] for kernel in kernels)
        coeffs = np.zeros((total_size, total_size))
        current_index = 0
        for kernel in kernels:
            size = kernel.coeffs.shape[0]
            coeffs[current_index:current_index+size,
                   current_index:current_index+size] = kernel.coeffs
            current_index += size
        return coeffs
    
    def eval_basis(self, index, kx, ky, kz):
        current_index = 0
        for kernel in self.kernels:
            size = kernel.coeffs.shape[0]
            if index < current_index + size:
                return kernel.eval_basis(index - current_index, kx, ky, kz)
            current_index += size
        raise IndexError("Index out of range for the combined basis.")


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


class VonMisesKernel(ScatteringKernel):
    """Scattering kernel based on von Mises distributions in the angle
    of the wavevector in the x-y plane.
    
    The basis consists of a constant term and one function of the form:
    
    .. math::
        e^{\\kappa \\cos(m(\\phi-\\phi_0))}
    
    So, the full kernel is:

    .. math::
        C(\\phi, \\phi') = C_0 + C_1 (
        e^{\\kappa \\cos(m(\\phi-\\phi_0))}
        e^{\\kappa \\cos(m(\\phi'-\\phi_0))})

    Parameters
    ----------
    params : dict
        A dictionary containing the following keys:
        * | ``c0``: The constant term in the kernel.
        * | ``c1``: The coefficient for the von Mises distribution.
        * | ``kappa``: Controls the width of the distribution. Larger
          | kappa corresponds to a narrower distribution.
        * | ``m``: Sets the symmetry of the distribution over the
          | angle in the x-y plane. For example, ``m=4`` repeats the
          | peak every 90 degrees.
        * | ``phi_0``: Sets the angle at which the peak of the
          | distribution occurs. Should be in degrees.
    """
    def __init__(self, params: Mapping):
        super().__init__(params)
    
    def build_coeffs(self, params: Mapping):
        self.coeffs = np.array([[params['c0'], params['c1']],
                                [params['c1'], 0]])
        self._phi0_rad = np.radians(params['phi_0'])
        return self.coeffs
    
    def eval_basis(self, index, kx, ky, kz):
        phi = np.arctan2(ky, kx)
        if index == 0:
            return 1
        else:
            exponent = np.cos(self.params['m'] * (phi - self._phi0_rad))
            return np.exp(self.params['kappa'] * exponent)


class CustomKernel(ScatteringKernel):
    """Scattering kernel expanded by the eigenvalue decomposition of
    a custom kernel function.

    Parameters
    ----------
    params : dict
        A dictionary containing the following keys:
        * | ``kernel_func``: A callable that takes in wavevector
          | components (kx, ky, kz, kx', ky', kz') and returns the value
          | of the kernel function at that wavevector. The output should
          | be in units of angstrom^2 THz. The wavevector components
          | have units of 1/angstrom.
        * | ``rank``: The number of eigenvalue and eigenvector pairs to
          | keep when decomposing the kernel function into basis
          | functions and coefficients. This would be the final rank of
          | the resulting scattering kernel. If not provided, set to 20
          | by default.
        * | ``low_res``: The resolution of the approximate band object
          | used for a more managable eigenvalue decomposition
          | (nystrom method). If not provided, set to 31 by default.
        * | ``decomp_method``: The method to use for the eigenvalue
          | decomposition. Should be one of 'lanczos' or 'full'. If not
          | provided, set to 'lanczos' by default. 'lanczos' should be
          | more efficient for larger low-res approximations of
          | the kernel.
    """
    def __init__(self, params: Mapping):
        self.params = params
        self.is_explicit = False
        self.coeffs = None
    
    def decompose(self, band):
        band_decomp = copy(band)
        band_decomp.resolution = self.params.get('low_res', 31)
        band_decomp.discretize()
        low_to_high = _interpolation_matrix(band.kpoints, band_decomp.kpoints)
        high_to_low = _interpolation_matrix(band_decomp.kpoints, band.kpoints)

        kx, ky, kz = band_decomp.kpoints.T
        kx, ky, kz = kx[:, None], ky[:, None], kz[:, None]
        kx_prime, ky_prime, kz_prime = kx.T, ky.T, kz.T
        kernel_matrix = self.params['kernel_func'](
            kx, ky, kz, kx_prime, ky_prime, kz_prime)

        method = self.params.get('decomp_method', 'lanczos')
        if method == 'full':
            eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
            cutoff = np.arange(
                0, min(self.params.get('rank', 20), len(eigenvalues)))
            eigenvalues = eigenvalues[-cutoff::-1]
            eigenvectors = eigenvectors[:, -cutoff::-1]
        elif method == 'lanczos':
            matvec = scipy.sparse.linalg.LinearOperator(
                kernel_matrix.shape, matvec=lambda x: kernel_matrix @ x)
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                matvec, k=self.params.get('rank', 20))
        
        self.coeffs = np.diag(eigenvalues)
        self.eigenvectors = eigenvectors
        self.projector = low_to_high @ eigenvectors
        self.inv_projector = eigenvectors.T @ high_to_low
        return self.coeffs
    
    def eval_basis(self, index, kx, ky, kz):
        return self.projector[index, :]


class IsotropicKernelFunction:
    """Scattering kernel that is just a constant function of the wavevectors
    of the incoming and outgoing states. This corresponds to isotropic
    scattering.

    Call the object like ``C(kx, ky, kz, kx_prime, ky_prime, kz_prime)``
    to evaluate the kernel function at the given wavevectors.

    Parameters
    ----------
    C : float
        The value of the kernel function at all wavevectors. Should be in
        units of angstrom^2 THz.
    """
    def __init__(self, C):
        self.C = C

    def __call__(self, kx, ky, kz, kx_prime, ky_prime, kz_prime):
        return self.C * np.ones((kx.shape[0], kx_prime.shape[1]))


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
        The width of the Gaussian.
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
        sign = 1 if self.backward else -1
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
    C(\\phi, \\phi') = (C_0 + C_1 \\mathrm{cos}\\left(
        \\frac{m(\\phi+\\phi_0)}{2}\\right)) \\mathrm{exp}\\left(
        -\\frac{(|\\phi-\\phi'|-\\delta)^2} {2(\\sigma_0+\\sigma_1
        \\mathrm{cos}\\left(\\frac{m(\\phi-\\phi_0)}{2}\\right))^2}
        \\right)

    Call the object like ``C(kx, ky, kz, kx_prime, ky_prime, kz_prime)``
    to evaluate the kernel function at the given wavevectors.

    Parameters
    ----------
    C0 : float
        The constant term in the amplitude of the Gaussian.
    C1 : float
        The coefficient for the anisotropic term in the amplitude of the
        Gaussian.
    sigma0 : float
        The constant term in the width of the Gaussian.
    sigma1 : float
        The coefficient for the anisotropic term in the width of the
        Gaussian.
    m : int
        Sets the symmetry of the anisotropy over the angle in the x-y
        plane. For example, ``m=4`` repeats the peak every 90 degrees.
    phi0 : float
        Sets the angle at which the peak of the anisotropy occurs.
        Should be in degrees.
    delta : float
        The shift of the Gaussian from zero, in degrees. This can be
        used to model scattering that is peaked at a non-zero angle
        difference, such as forward scattering (``delta=0``) or
        backward scattering (``delta=180``).
    """
    def __init__(self, C0, C1, sigma0, sigma1, m, phi0, delta=0.0):
        self.C0 = C0
        self.C1 = C1
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.m = m
        self.phi0_rad = np.radians(phi0)
        self.delta_rad = np.radians(delta)
    def __call__(self, kx, ky, kz, kx_prime, ky_prime, kz_prime):
        phi = np.arctan2(ky, kx)
        phi_prime = np.arctan2(ky_prime, kx_prime)
        amplitude = self.C0 + self.C1 * np.cos(
            self.m * (phi - self.phi0_rad))
        width = self.sigma0 + self.sigma1 * np.cos(
            self.m * (phi - self.phi0_rad))
        diff_phi = abs(phi-phi_prime) - self.delta_rad
        return amplitude * np.exp(-diff_phi**2 / (2 * width**2))


class SumKernelFunction:
    """Class for gathering multiple kernels together by adding
    their kernel functions.
    
    Call the object like ``C(kx, ky, kz, kx_prime, ky_prime, kz_prime)``
    to evaluate the combined kernel function at the given wavevectors.

    Parameters
    ----------
    kernels : list of Callable
        A list of kernel functions to add together.
    """
    def __init__(self, kernels: Collection[Callable]):
        self.kernels = kernels
    def __call__(self, kx, ky, kz, kx_prime, ky_prime, kz_prime):
        result = 0
        for kernel in self.kernels:
            result += kernel(kx, ky, kz, kx_prime, ky_prime, kz_prime)
        return result


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
          | :math:`C_f \\mathrm{exp}\\left(\\frac{-|\\phi-\\phi'|^2}
          | {2\\sigma_f^2}\\right)`.
        * | ``'backward_phi'``: Like ``'backward'``, but the Gaussian
          | is in the angle of the wavevector in the x-y plane instead
          | of the full wavevector. So,
          | :math:`C_b \\mathrm{exp}\\left(\\frac{-|\\phi+\\phi'|^2}
          | {2\\sigma_b^2}\\right)`.
        * | ``'forward_anisotropic'``: Like ``'forward_phi'``, but with
          | anisotropic parameters for the Gaussian. This means,
          | :math:`C_f = C_{f0} + C_{f1}
          | \\mathrm{cos}\\left(\\frac{m(\\phi-\\phi_0)}{2}\\right)` and
          | :math:`\\sigma_f=\\sigma_{f0}+\\sigma_{f1}
          | \\mathrm{cos}\\left(\\frac{m(\\phi+\\phi_0)}{2}\\right)`.
          | The kernel parameters are ``'Cf0'``, ``'Cf1'``,
          | ``'sigmaf0'``, ``'sigmaf1'``, ``'m'``,
          | and ``'phi0'``.
        * | ``'backward_anisotropic'``: Like ``'backward_phi'``, but
          | with anisotropic parameters for the Gaussian. This means,
          | :math:`C_b = C_{b0} + C_{b1}
          | \\mathrm{cos}\\left(\\frac{m(\\phi+\\phi_0)}{2}\\right)` and
          | :math:`\\sigma_b=\\sigma_{b0}+\\sigma_{b1}
          | \\mathrm{cos}\\left(\\frac{m(\\phi+\\phi_0)}{2}\\right)`.
          | The kernel parameters are ``'Cb0'``, ``'Cb1'``,
          | ``'sigmab0'``, ``'sigmab1'``, ``'m'``,
          | and ``'phi0'``.

        If a list of kernels is provided, the resulting kernel is the
        sum of all the kernels in the list. Note that you cannot mix
        kernels with explicit basis functions and kernels without
        explicit basis functions.

    kernel_params : dict or list of dict
        The parameters for the kernel(s). If a list of explicit basis
        kernels is provided, a list of parameter dictionaries should
        be provided, where each dictionary corresponds to the
        parameters for the respective kernel. If the kernels do not
        have explicit basis functions, as long as the parameter names
        do not conflict, you can just provide a single dictionary with
        all the parameters for all the kernels. If not, a list can
        still be provided, where each dictionary corresponds to the
        parameters for the respective kernel. For non-explicit kernels,
        an extra dictionary can be provided at the end of the list to
        specify the parameters for the decomposition of the resulting
        kernel (see `elecboltz.kernel.CustomKernel`).
    """
    if isinstance(kernel, list):
        is_explicit = is_kernel_name_for_explicit(kernel[0])
        for k in kernel:
            if is_kernel_name_for_explicit(k) != is_explicit:
                raise ValueError("Cannot mix kernels with and without "
                                 "explicit basis functions.")
        decomp_params = None
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
        if is_explicit:
            return SumKernel(kernels)
        else:
            return CustomKernel(
                {'kernel_func': SumKernelFunction(
                    k.params['kernel_func'] for k in kernels),
                 **(decomp_params or {})})
    elif kernel == 'isotropic':
        return CustomKernel({'kernel_func': IsotropicKernelFunction(
            kernel_params['C0']), **kernel_params})
    elif kernel == 'forward':
        return CustomKernel({'kernel_func': GaussianScattering(
            kernel_params['Cf'], kernel_params['sigmaf']), **kernel_params})
    elif kernel == 'backward':
        return CustomKernel({'kernel_func': GaussianScattering(
            kernel_params['Cb'], kernel_params['sigmab'], backward=True),
            **kernel_params})
    elif kernel == 'forward_phi':
        return CustomKernel({'kernel_func': AnisotropicGaussianScattering(
            kernel_params['Cf'], 0, kernel_params['sigmaf'], 0, 1, 0),
            **kernel_params})
    elif kernel == 'backward_phi':
        return CustomKernel({'kernel_func': AnisotropicGaussianScattering(
            kernel_params['Cb'], 0, kernel_params['sigmab'], 0, 1, 0,
            delta=180), **kernel_params})
    elif kernel == 'forward_anisotropic':
        return CustomKernel({'kernel_func': AnisotropicGaussianScattering(
            kernel_params['Cf0'], kernel_params['Cf1'],
            kernel_params['sigmaf0'], kernel_params['sigmaf1'],
            kernel_params['m'], kernel_params['phi0']),
            **kernel_params})
    elif kernel == 'backward_anisotropic':
        return CustomKernel({'kernel_func': AnisotropicGaussianScattering(
            kernel_params['Cb0'], kernel_params['Cb1'],
            kernel_params['sigmab0'], kernel_params['sigmab1'],
            kernel_params['m'], kernel_params['phi0'], delta=180),
            **kernel_params})
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
    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")


def is_kernel_name_for_explicit(kernel_name):
    return kernel_name in ['spherical', 'cylindrical', 'legendre']


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
