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


class SphericalHarmonicsKernel(ScatteringKernel):
    """Scattering kernel based on spherical harmonics
    :math:`Y^l_m(\\theta, \\phi)`.

    Parameters
    ----------
    params : dict
        A dictionary mapping tuples of tuples of integers,
        ((l, m), (l', m')), to the corresponding coefficients of
        the scattering kernel. Note that since the scattering kernel
        should be Hermitian, you must only specify one coefficient for
        each pair of indices. The kernel will automatically fill in the
        other coefficient.
    """
    def __init__(self, params):
        super().__init__(params)

    def build_coeffs(self, params):
        matrix_dict = {}
        for (l, m), (l_prime, m_prime) in params['coeffs'].keys():
            matrix_dict[(l*(l+1) + m, l_prime*(l_prime+1) + m_prime)] = \
                params['coeffs'][((l, m), (l_prime, m_prime))]
        matrix_size = max(max(k) for k in matrix_dict.keys()) + 1
        self.coeffs = np.zeros((matrix_size, matrix_size), dtype=complex)
        for (i, j), value in matrix_dict.items():
            self.coeffs[i, j] = value
        self.coeffs += self.coeffs.conj().T - np.diag(self.coeffs.diagonal())
        return self.coeffs

    def eval_basis(self, index, kx, ky, kz):
        theta = np.arccos(kz / np.sqrt(kx**2 + ky**2 + kz**2))
        phi = np.arctan2(ky, kx)
        l = index // (index + 1)
        m = index % (index + 1)
        return sph_harm_y(l, m, theta, phi)


class CylindricalHarmonicsKernel(ScatteringKernel):
    """Scattering kernel based on cylindrical harmonics
    :math:`e^{im\\phi}`.

    Parameters
    ----------
    params : dict or np.ndarray
        Either a dictionary mapping (m, m') to the non-zero
        coefficients of the scattering kernel, or a 2D array of
        coefficients where the entry at (m, m') corresponds to the
        coefficient for the basis functions with indices m and m'.
        Note that since the scattering kernel should be Hermitian,
        if you use a dictionary, you must only specify one coefficient
        for each pair of indices. The kernel will automatically fill in
        the other coefficient.
    """
    def __init__(self, params):
        super().__init__(params)

    def build_coeffs(self, params):
        if isinstance(params['coeffs'], dict):
            matrix_dict = {}
            for (m, m_prime), value in params['coeffs'].items():
                matrix_dict[(m, m_prime)] = value
            matrix_size = max(max(k) for k in matrix_dict.keys()) + 1
            self.coeffs = np.zeros((matrix_size, matrix_size), dtype=complex)
            for (i, j), value in matrix_dict.items():
                self.coeffs[i, j] = value
        else:
            self.coeffs = params['coeffs']
        self.coeffs += self.coeffs.conj().T - np.diag(self.coeffs.diagonal())
        return self.coeffs

    def eval_basis(self, index, kx, ky, kz):
        phi = np.arctan2(ky, kx)
        return np.exp(1j * index * phi)


class LegendrePolynomialsKernel(ScatteringKernel):
    """Scattering kernel based on Legendre polynomials
    :math:`P_l(\\cos \\theta)`.

    To preserve normalization, this kernel actually uses the spherical
    harmonics :math:`Y^l_0(\\theta, \\phi)`, which are proportional to
    the Legendre polynomials :math:`P_l(\\cos \\theta)`.

    Parameters
    ----------
    params : dict
        Either a dictionary mapping (l, l') to the non-zero
        coefficients of the scattering kernel, or a 2D array of
        coefficients where the entry at (l, l') corresponds to the
        coefficient for the basis functions with indices l and l'.
        Note that since the scattering kernel should be Hermitian,
        if you use a dictionary, you must only specify one coefficient
        for each pair of indices. The kernel will automatically fill in
        the other coefficient.
    """
    def __init__(self, params):
        super().__init__(params)

    def build_coeffs(self, params):
        if isinstance(params['coeffs'], dict):
            matrix_dict = {}
            for (l, l_prime), value in params['coeffs'].items():
                matrix_dict[(l, l_prime)] = value
            matrix_size = max(max(k) for k in matrix_dict.keys()) + 1
            self.coeffs = np.zeros((matrix_size, matrix_size), dtype=complex)
            for (i, j), value in matrix_dict.items():
                self.coeffs[i, j] = value
        else:
            self.coeffs = params['coeffs']
        self.coeffs += self.coeffs.conj().T - np.diag(self.coeffs.diagonal())
        return self.coeffs

    def eval_basis(self, index, kx, ky, kz):
        theta = np.arccos(kz / np.sqrt(kx**2 + ky**2 + kz**2))
        return sph_harm_y(index, 0, theta, 0)
