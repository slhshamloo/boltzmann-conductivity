import unittest
import elecboltz
import numpy as np

class TestGeometricalCalculations(unittest.TestCase):
    def setUp(self):
        # make dummy bandstructure and conductivity objects to inject
        # testing coordinates into them later for just testing the 
        # jacobian calculations
        self.band = elecboltz.BandStructure(
            "Ef * (kx^2 + ky^2 + kz^2)", 1.0, [np.pi, np.pi, np.pi],
            band_params={'Ef': 1.0})
        self.cond = elecboltz.Conductivity(
            self.band, field=[0.0, 0.0, 0.0], scattering_rate=1.0)
    
    def test_regular_octahedron_jacobian(self):
        self.set_up_regular_octahedron()
        self.cond._calculate_jacobian_sums(self.triangle_coordinates)
        neighbor_pairs = {
            (0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4),
            (2, 4), (3, 4), (0, 5), (1, 5), (2, 5), (3, 5)}
        for i in range(6):
            for j in range(6):
                if i == j:
                    value = 4 * np.sqrt(3)
                elif (i, j) in neighbor_pairs or (j, i) in neighbor_pairs:
                    value = 2 * np.sqrt(3)
                else:
                    value = 0.0
                self.assertAlmostEqual(
                    self.cond._jacobian_sums[i, j], value, 5,
                    f"Regular octahedron Jacobian sum at ({i}, {j})"
                    "is incorrect")

    def test_regular_octahedron_derivative(self):
        self.set_up_regular_octahedron()
        self.cond._calculate_derivative_sums(self.triangle_coordinates)
        derivative_sums = np.zeros((6, 6, 3))
        derivative_sums[0, 1, 2] = -2
        derivative_sums[1, 0, 2] = 2
        derivative_sums[1, 2, 2] = -2
        derivative_sums[2, 1, 2] = 2
        derivative_sums[2, 3, 2] = -2
        derivative_sums[3, 2, 2] = 2
        derivative_sums[3, 0, 2] = -2
        derivative_sums[0, 3, 2] = 2
        derivative_sums[0, 4, 0] = 2
        derivative_sums[4, 0, 0] = -2
        derivative_sums[1, 4, 1] = 2
        derivative_sums[4, 1, 1] = -2
        derivative_sums[2, 4, 0] = -2
        derivative_sums[4, 2, 0] = 2
        derivative_sums[3, 4, 1] = -2
        derivative_sums[4, 3, 1] = 2
        derivative_sums[0, 5, 0] = -2
        derivative_sums[5, 0, 0] = 2
        derivative_sums[1, 5, 1] = -2
        derivative_sums[5, 1, 1] = 2
        derivative_sums[2, 5, 0] = 2
        derivative_sums[5, 2, 0] = -2
        derivative_sums[3, 5, 1] = 2
        derivative_sums[5, 3, 1] = -2
        for i in range(6):
            for j in range(6):
                for k in range(3):
                    self.assertEqual(
                        self.cond._derivatives[k][i, j],
                        derivative_sums[i, j, k],
                        "Regular octahedron derivative sum at"
                        f" ({i}, {j}) is incorrect on axis {k}")

    def set_up_regular_octahedron(self):
        vertices = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        faces = np.array([
            [0, 1, 4], [0, 5, 1], [1, 2, 4], [1, 5, 2],
            [2, 3, 4], [2, 5, 3], [3, 0, 4], [3, 5, 0]])
        self.set_up_geometry(vertices, faces)

    def set_up_geometry(self, vertices, faces):
        self.band.kpoints = vertices
        self.band.kpoints_periodic = vertices.copy()
        self.band.kfaces = faces
        self.band.kfaces_periodic = faces.copy()
        self.triangle_coordinates = self.band.kpoints[self.band.kfaces]
        self.cond._bandwidth = np.max(np.abs(
            self.band.kfaces - np.roll(self.band.kfaces, 1, axis=1)))


if __name__ == "__main__":
    unittest.main()
