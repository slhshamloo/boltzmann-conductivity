import unittest
import elecboltz
import tempfile
import pathlib
import numpy as np


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        with open(pathlib.Path(self.temp_dir.name)
                  / 'test_data_phi30_B_10.0_rho0=0.5_date_11072025.csv',
                  'w') as f:
            f.write("# these\n"
                    "# are\n"
                    "# comments\n"
                    "theta,rho_a,rho_c\n"
                    "0.5,0.01,0.1\n"
                    "1.0,0.02,0.2\n"
                    "1.5,0.03,0.3")
        with open(pathlib.Path(self.temp_dir.name)
                  / 'test_data_phi=45_B2.4_rho0=1.0_date12072025.csv',
                  'w') as f:
            f.write("# more\n"
                    "# comments\n"
                    "# here\n"
                    "theta,rho_a,rho_c\n"
                    "2.0,0.04,0.4\n"
                    "2.5,0.05,0.5\n"
                    "3.0,0.06,0.6\n"
                    "3.5,0.07,0.7\n"
                    "4.0,0.08,0.8\n")
        with open(pathlib.Path(self.temp_dir.name)
                  / 'test_data_phi=15_B3.0_rho0=1.5_date=13072025.csv',
                  'w') as f:
            f.write("# even\n"
                    "# more\n"
                    "# comments\n"
                    "theta,rho_a,rho_c\n"
                    "4.5,0.09,0.9\n"
                    "5.0,0.1,1.0\n")
        with open(pathlib.Path(self.temp_dir.name)
                  / 'not_the_data_phi=15_B3.0_rho0=1.5.csv', 'w') as f:
            f.write("# yet\n"
                    "# more\n"
                    "# comments\n"
                    "theta,rho_a,rho_c\n"
                    "5.5,0.11,1.1\n"
                    "6.0,0.12,1.2\n")

    def test_auto_search(self):
        loader = elecboltz.Loader(
            data_type='plain', extra_labels=['rho0', 'date'])
        loader.load(folder_path=str(self.temp_dir.name),
                    prefix='test_data_', header_lines=3)
        self.assertTrue(all(
            val == test_val for val, test_val
            in zip(loader.extra_values[0], [0.5, 1.5, 1.0])),
            "Extra values do not match expected values.")
        self.assertTrue(all(
            np.all(x == x_test) for x, x_test in
            zip(loader.x_data_raw, [np.array([0.5, 1.0, 1.5]),
                                    np.array([4.5, 5.0]),
                                    np.array([2.0, 2.5, 3.0, 3.5, 4.0])])),
            "Raw x data does not match expected values.")
        self.assertTrue(all(
            np.all(y == y_test) for y, y_test in
            zip(loader.y_data_raw[0],
                [np.array([0.01, 0.02, 0.03]),
                 np.array([0.09, 0.1]),
                 np.array([0.04, 0.05, 0.06, 0.07, 0.08])])),
            "Raw y data does not match expected values.")
        self.assertTrue(all(
            np.all(y == y_test) for y, y_test in
            zip(loader.y_data_raw[1], [np.array([0.1, 0.2, 0.3]),
                                       np.array([0.9, 1.0]),
                                       np.array([0.4, 0.5, 0.6, 0.7, 0.8])])),
            "Raw y data does not match expected values.")
        self.assertTrue(all(
            label == label_test for label, label_test in
            zip(loader.x_label, ['theta', 'phi', 'B'])),
            "x labels do not match expected values.")
        self.assertTrue(all(
            np.all(x == x_test) for x, x_test in zip(loader.x_data,
                [np.array([0.5, 1.0, 1.5, 4.5, 5.0, 2.0, 2.5, 3.0, 3.5, 4.0]),
                 np.array([30.0, 30.0, 30.0, 15.0, 15.0,
                           45.0, 45.0, 45.0, 45.0, 45.0]),
                 np.array([10.0, 10.0, 10.0, 3.0, 3.0,
                           2.4, 2.4, 2.4, 2.4, 2.4])])),
            "x data does not match expected values.")
        self.assertTrue(all(
            np.all(y == y_test) for y, y_test in zip(loader.y_data,
                [np.array([0.01, 0.02, 0.03, 0.09, 0.1,
                           0.04, 0.05, 0.06, 0.07, 0.08]),
                 np.array([0.1, 0.2, 0.3, 0.9, 1.0,
                           0.4, 0.5, 0.6, 0.7, 0.8])])),
            "y data does not match expected values.")
    
    def test_plain_load(self):
        loader = elecboltz.Loader(
            x_vary_label='theta', x_labels_search=['phi', 'B'],
            x_values_search=[[30, 45], [10.0, 2.4]],
            y_label=['rho_xx', 'rho_xy'], extra_labels=['rho0'],
            data_type='plain')
        loader.load(folder_path=str(self.temp_dir.name),
                    prefix='test_data_', y_columns=[1, 2],
                    header_lines=3)
        self.assertTrue(all(
            val == test_val for val, test_val
            in zip(loader.extra_values[0], [0.5, 1.0])),
            "Extra values do not match expected values.")
        self.assertTrue(all(
            np.all(x == x_test) for x, x_test in
            zip(loader.x_data_raw,[np.array([0.5, 1.0, 1.5]),
                                   np.array([2.0, 2.5, 3.0, 3.5, 4.0])])),
            "Raw x data does not match expected values.")
        self.assertTrue(all(
            np.all(y == y_test) for y, y_test in
            zip(loader.y_data_raw[0],
                [np.array([0.01, 0.02, 0.03]),
                 np.array([0.04, 0.05, 0.06, 0.07, 0.08])])),
            "Raw y data does not match expected values.")
        self.assertTrue(all(
            np.all(y == y_test) for y, y_test in
            zip(loader.y_data_raw[1], [np.array([0.1, 0.2, 0.3]),
                                       np.array([0.4, 0.5, 0.6, 0.7, 0.8])])),
            "Raw y data does not match expected values.")
        self.assertTrue(all(
            label == label_test for label, label_test in
            zip(loader.x_label, ['theta', 'phi', 'B'])),
            "x labels do not match expected values.")
        self.assertTrue(all(
            np.all(x == x_test) for x, x_test in zip(loader.x_data,
                [np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
                 np.array([30.0, 30.0, 30.0, 45.0, 45.0, 45.0, 45.0, 45.0]),
                 np.array([10.0, 10.0, 10.0, 2.4, 2.4, 2.4, 2.4, 2.4])])),
            "x data does not match expected values.")
        self.assertTrue(all(
            np.all(y == y_test) for y, y_test in zip(loader.y_data,
                [np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
                 np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])])),
            "y data does not match expected values.")

    def test_admr_load(self):
        loader = elecboltz.Loader(
            x_vary_label='theta', x_labels_search=['phi', 'B'],
            x_values_search=[[30, 45], [10.0, 2.4]], data_type='admr')
        loader.load(folder_path=str(self.temp_dir.name),
                    prefix='test_data_', y_columns=[1, 2],
                    header_lines=3)
        theta = np.deg2rad(np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]))
        phi = np.deg2rad(
            np.array([30.0, 30.0, 30.0, 45.0, 45.0, 45.0, 45.0, 45.0]))
        B = np.array([10.0, 10.0, 10.0, 2.4, 2.4, 2.4, 2.4, 2.4])
        self.assertEqual(loader.x_label, ['field'])
        self.assertTrue(np.all(loader.x_data[0] == B[:, None] * np.column_stack((
                np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                np.cos(theta)))),
            "x data does not match expected values.")


if __name__ == '__main__':
    unittest.main()
