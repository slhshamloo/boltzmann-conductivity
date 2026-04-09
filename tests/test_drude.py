import unittest

import elecboltz
import numpy as np
from scipy.constants import e, electron_volt, hbar, m_e


class TestDrude(unittest.TestCase):
    def calculate_drude(self, effective_mass, charge_density,
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

    def make_bandstructure(self, kf, three_dimensional=False, **kwargs):
        dispersion = "Ef * (kx^2 + ky^2"
        if three_dimensional:
            dispersion += " + kz^2"
        dispersion += ")"

        a = np.pi / kf * 1e10
        coeff = hbar**2 / (2 * m_e)
        ef = coeff * kf**2 / electron_volt * 1e3
        bandstructure = elecboltz.BandStructure(
            dispersion, ef, [a, a, a], band_params={"Ef": ef}, **kwargs)
        bandstructure.discretize()
        return bandstructure
    
    def calculate_conductivities(self, B):
        kf = 1e10
        bandstructure = self.make_bandstructure(kf, periodic=2, resolution=21)
        conductivity = elecboltz.Conductivity(
            bandstructure, scattering_rate=1.0, field=B)
        conductivity.calculate()
        sigma_boltzmann = conductivity.sigma

        charge_density = kf**3 / (2 * np.pi**2)
        sigma_drude = self.calculate_drude(
            1.0, charge_density, 1.0 * 1e12, B)
        
        return sigma_boltzmann, sigma_drude

    def test_zero_field(self):
        sigma_boltzmann, sigma_drude = self.calculate_conductivities(
            B=np.array([0.0, 0.0, 0.0]))

        relative_error_xx = abs(sigma_boltzmann[0, 0]-sigma_drude[0, 0]
                                ) / abs(sigma_drude[0, 0])
        self.assertLess(
            relative_error_xx, 0.01,
            f"xx relative error {relative_error_xx:.3g} exceeds 1% tolerance")

    def test_nonzero_field(self):
        sigma_boltzmann, sigma_drude = self.calculate_conductivities(
            B=np.array([0.0, 0.0, 10.0]))

        relative_error_xx = abs(sigma_boltzmann[0, 0]-sigma_drude[0, 0]
                                ) / abs(sigma_drude[0, 0])
        relative_error_xy = abs(sigma_boltzmann[0, 1]-sigma_drude[0, 1]
                                ) / abs(sigma_drude[0, 1])
        self.assertLess(
            relative_error_xx, 0.01,
            f"finite-field xx relative error {relative_error_xx:.3g}" \
            f" exceeds 1% tolerance",)
        self.assertLess(
            relative_error_xy, 0.01,
            f"xy relative error {relative_error_xy:.3g} exceeds 1% tolerance",)


if __name__ == "__main__":
    unittest.main()
