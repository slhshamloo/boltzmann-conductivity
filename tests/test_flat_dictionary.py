import unittest
from elecboltz.fit import _extract_flat_keys, _build_params_from_flat

class TestFlatDictionary(unittest.TestCase):
    def test_extract_flat_keys(self):
        params = {
            'energy_scale': (100.0, 200.0),
            'scattering_rate': (0.1, 10.0),
            'band_params': {
                'a': (0.1, 1.0),
                'b': (0.1, 1.0),
                'c': (0.1, 1.0),
            },
            'scattering_params': {
                'nu': [(0.1, 10.0), (0.1, 10.0)],
                'gamma': [(0.1, 10.0), (0.1, 10.0)],
            }
        }
        expected_keys = [
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
        self.assertEqual(_extract_flat_keys(params), expected_keys,
                         "Extracted flat keys do not match expected keys.")

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
