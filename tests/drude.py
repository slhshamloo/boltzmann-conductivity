import elecboltz
import numpy as np
from scipy.constants import e, electron_volt, hbar, m_e


def calculate_drude(effective_mass, charge_density,
                    scattering_rate, magnetic_field):
        charge_mass = m_e * effective_mass
        omega_c = e * magnetic_field / charge_mass
        inverse_tensor = np.zeros((3, 3))
        np.fill_diagonal(inverse_tensor, scattering_rate)
        inverse_tensor[0, 1] = omega_c[2]
        inverse_tensor[1, 0] = -omega_c[2]
        inverse_tensor[0, 2] = -omega_c[1]
        inverse_tensor[2, 0] = omega_c[1]
        inverse_tensor[1, 2] = omega_c[0]
        inverse_tensor[2, 1] = -omega_c[0]
        return charge_density * e**2 / charge_mass * np.linalg.inv(inverse_tensor)

def make_isotropic_bandstructure(kf, three_dimensional=False, **kwargs):
    dispersion = "Ef * (kx^2 + ky^2"
    if three_dimensional:
        dispersion += " + kz^2"
    dispersion += ") / kf^2"

    a = np.pi / kf * 1e10
    coeff = hbar**2 / (2 * m_e)
    ef = coeff * kf**2 / electron_volt * 1e3
    bandstructure = elecboltz.BandStructure(
        dispersion, ef, [a, a, a],
        band_params={'Ef': ef, 'kf': kf * 1e-10}, **kwargs)
    bandstructure.discretize()
    return bandstructure