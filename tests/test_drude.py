import unittest
import elecboltz
import numpy as np
from tests.drude import calculate_drude, make_isotropic_bandstructure


class TestDrude(unittest.TestCase):
    def calculate_conductivities(self, B):
        kf = 1e10
        bandstructure = make_isotropic_bandstructure(
            kf, periodic=2, resolution=21)
        conductivity = elecboltz.Conductivity(
            bandstructure, scattering_rate=1.0, field=B)
        conductivity.calculate()
        sigma_boltzmann = conductivity.sigma

        charge_density = kf**3 / (2 * np.pi**2)
        sigma_drude = calculate_drude(1.0, charge_density, 1.0 * 1e12, B)
        
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
