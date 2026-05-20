import re
import numpy as np
import scipy
if scipy.__version__ >= "1.15.0":
    from scipy.special import sph_harm_y
else:
    from scipy.special import sph_harm as sph_harm_y


class ScatteringKernel:
    def __init__(self, params):
        self.coeffs = self.build_coeffs(params)
    
    def build_coeffs(self, params) -> np.ndarray:
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
    def __init__(self, kernels):
        self.kernels = kernels
        self.coeffs = self.build_coeffs(kernels)
    
    def build_coeffs(self, kernels):
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
    def __init__(self, params):
        super().__init__(params)

    def build_coeffs(self, params):
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
    def __init__(self, params):
        super().__init__(params)

    def build_coeffs(self, params):
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
    def __init__(self, params):
        super().__init__(params)

    def build_coeffs(self, params):
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


def build_kernel(kernel, kernel_params):
    """Build a scattering kernel based on the given kernel type and
    parameters.

    Parameters
    ----------
    kernel : str or list of str
        The type(s) of kernel(s) to build. Supported kernels are:

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

        If a list of kernels is provided, the resulting kernel will be
        a sum of the individual kernels.
    kernel_params : dict or list of dict
        The parameters for the kernel(s). If a list of kernels is
        provided, a list of parameter dictionaries should be provided,
        where each dictionary corresponds to the parameters for the
        respective kernel.
    """
    if isinstance(kernel, list):
        kernels = []
        for k, p in zip(kernel, kernel_params):
            kernels.append(build_kernel(k, p))
        return SumKernel(kernels)
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
