import unittest
import elecboltz
import numpy as np
from scipy.constants import angstrom, angstrom, e, electron_volt, hbar, m_e


class TestDrude(unittest.TestCase):
    def calculate_conductivities(self, B, scattering=1.0, kf=1):
        kf /= angstrom
        bandstructure = make_isotropic_bandstructure(
            kf, periodic=2, resolution=21)
        conductivity = elecboltz.Conductivity(
            bandstructure, scattering_rate=scattering, field=B)
        conductivity.calculate()
        sigma_rta = conductivity.sigma

        kernel = elecboltz.kernel.LegendreKernel(
            {(0, 0): scattering / (kf * angstrom)**2})
        conductivity = elecboltz.Conductivity(
            bandstructure, scattering_kernel=kernel, field=B)
        conductivity.calculate()
        sigma_kernel = conductivity.sigma


        charge_density = kf**3 / (2 * np.pi**2)
        sigma_drude = calculate_drude(1.0, charge_density, 1.0 * 1e12, B)
        
        return sigma_kernel, sigma_rta, sigma_drude

    def test_zero_field(self):
        sigma_kernel, sigma_rta, sigma_drude = self.calculate_conductivities(
            B=np.array([0.0, 0.0, 0.0]))

        error_kernel = abs(sigma_kernel[0, 0]-sigma_drude[0, 0]
                           ) / abs(sigma_drude[0, 0])
        error_rta = abs(sigma_rta[0, 0]-sigma_drude[0, 0]
                        ) / abs(sigma_drude[0, 0])
        self.assertLess(
            error_kernel, 0.01,
            f"Scattering kernel conductivity error {error_kernel:.3g}"
            f" exceeds 1% tolerance")
        self.assertLess(
            error_rta, 0.01,
            f"RTA conductivity error {error_rta:.3g} exceeds 1% tolerance")

    def test_nonzero_field(self):
        sigma_kernel, sigma_rta, sigma_drude = self.calculate_conductivities(
            B=np.array([0.0, 0.0, 10.0]))

        error_xx_kernel = abs(sigma_kernel[0, 0]-sigma_drude[0, 0]
                              ) / abs(sigma_drude[0, 0])
        error_xy_kernel = abs(sigma_kernel[0, 1]-sigma_drude[0, 1]
                              ) / abs(sigma_drude[0, 1])
        error_xx_rta = abs(sigma_rta[0, 0]-sigma_drude[0, 0]
                           ) / abs(sigma_drude[0, 0])
        error_xy_rta = abs(sigma_rta[0, 1]-sigma_drude[0, 1]
                           ) / abs(sigma_drude[0, 1])
        self.assertLess(
            error_xx_kernel, 0.01,
            f"Scattering kernel conductivity xx error {error_xx_kernel:.3g}"
            f" exceeds 1% tolerance",)
        self.assertLess(
            error_xy_kernel, 0.01,
            f"Scattering kernel conductivity xy relative error"
            f"{error_xy_kernel:.3g} exceeds 1% tolerance")
        self.assertLess(
            error_xx_rta, 0.01,
            f"RTA conductivity xx error {error_xx_rta:.3g}"
            f"exceeds 1% tolerance")
        self.assertLess(
            error_xy_rta, 0.01,
            f"RTA conductivity xy relative error {error_xy_rta:.3g}"
            f"exceeds 1% tolerance")


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


if __name__ == "__main__":
    unittest.main()
