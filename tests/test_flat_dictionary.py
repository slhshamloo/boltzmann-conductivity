import unittest
from elecboltz.fit import _extract_flat_keys, _build_params_from_flat

class TestFlatDictionary(unittest.TestCase):
    def test_extract_flat_keys(self):
        params = {
            'energy_scale': 100.0,
            'scattering_rate': 1.0,
            'band_params': {
                'a': 2.0,
                'b': 3.0,
                'c': 5.0,
            },
            'scattering_models': ['isotropic', 'cos2phi'],
            'scattering_params': {
                'power': [6, 7],
                'gamma_0': [8.1, 9.2],
                'gamma_k': [52.1, 63.4]
            }
        }
        expected_keys = [
            'energy_scale', 
            'scattering_rate', 
            'band_params.a', 
            'band_params.b', 
            'band_params.c',
            'scattering_models.0',
            'scattering_models.1',
            'scattering_params.power.0', 
            'scattering_params.power.1',
            'scattering_params.gamma_0.0',
            'scattering_params.gamma_0.1',
            'scattering_params.gamma_k.0',
            'scattering_params.gamma_k.1'
        ]
        self.assertEqual(
            _extract_flat_keys(params, bounds=False), expected_keys,
            "Extracted flat keys do not match expected keys "
            "with bounds=False.")

        params = {
            'energy_scale': (100.0, 200.0),
            'scattering_rate': (0.1, 10.0),
            'band_params': {
                'a': (0.1, 1.0),
                'b': (0.1, 1.0),
                'c': (0.1, 1.0),
            },
            'scattering_params': {
                'power': [(1, 20), (3, 15)],
                'gamma_0': [(1.0, 10.0), (1.0, 10.0)],
                'gamma_k': [(1.0, 10.0), (1.0, 10.0)],
            }
        }
        expected_keys = [
            'energy_scale', 
            'scattering_rate', 
            'band_params.a', 
            'band_params.b', 
            'band_params.c',
            'scattering_params.power.0', 
            'scattering_params.power.1',
            'scattering_params.gamma_0.0',
            'scattering_params.gamma_0.1',
            'scattering_params.gamma_k.0',
            'scattering_params.gamma_k.1'
        ]
        self.assertEqual(
            _extract_flat_keys(params, bounds=True), expected_keys,
            "Extracted flat keys do not match expected keys "
            "with bounds=True.")

    def test_build_params_from_flat(self):
        keys = [
            'energy_scale', 
            'scattering_rate', 
            'band_params.a', 
            'band_params.b', 
            'band_params.c', 
            'scattering_params.nu.0', 
            'scattering_params.nu.1', 
            'scattering_params.gamma.0', 
            'scattering_params.gamma.1'
        ]
        values = [100.0, 1.0, 0.1, 0.2, 0.3, 4.0, 5.0, 6.0, 7.0]
        expected_params = {
            'energy_scale': 100.0,
            'scattering_rate': 1.0,
            'band_params': {
                'a': 0.1,
                'b': 0.2,
                'c': 0.3
            },
            'scattering_params': {
                'nu': [4.0, 5.0],
                'gamma': [6.0, 7.0]
            }
        }
        self.assertEqual(
            _build_params_from_flat(keys, values), expected_params,
            "Dictionary built from flat keys"
            "do not match expected parameters.")


if __name__ == '__main__':
    unittest.main()
