import unittest
import elecboltz


class TestChemicalPotentialTuning(unittest.TestCase):
    def test_ndlsco_tunes_to_p016(self):
        params = {
            'a': 3.75,
            'b': 3.75,
            'c': 13.2,
            'energy_scale': 160,
            'band_params': {
                'mu': -0.82439881,
                't': 1,
                'tp': -0.13642799,
                'tpp': 0.06816836,
                'tz': 0.06512192,
            },
            'resolution': 21,
            'domain_size': [1.0, 1.0, 2.0],
            'periodic': 2,
        }
        params = elecboltz.easy_params(params)
        band = elecboltz.BandStructure(
            params['dispersion'], params['chemical_potential'],
            params['unit_cell'], band_params=params['band_params'],
            fixed_filling=0.84, domain_size=params['domain_size'],
            periodic=params['periodic'], resolution=params['resolution'])

        band.discretize()
        self.assertAlmostEqual(band.n, 0.84, delta=0.02)


if __name__ == '__main__':
    unittest.main()