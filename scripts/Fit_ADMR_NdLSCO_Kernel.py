import elecboltz
import os
import pathlib
import numpy as np
from copy import copy


def get_init_params():
    return {
    'a': 3.75,
    'b': 3.75,
    'c': 13.2,
    'energy_scale': 160,
    'band_params': {'mu': -0.82439881, 't': 1, 'tp': -0.13642799,
                    'tpp': 0.06816836, 'tz': 0.06512192},
    'resolution': 41,
    'periodic': 2,
    'domain_size': [1.0, 1.0, 2.0],
    'Bamp': 45,
    'scattering_kernel_names': ['isotropic', 'hotspot_phi'],
    'scattering_kernel_params': [
        {'C_0': 2.9},
        {'phi_h': [0, 90, 180, 270], 'dphi_h': 90, 'C_h': 85, 'sigma_h': 0.13},
        {'rank': 20, 'low_res': 21}],
}

bounds = {
    'scattering_kernel_params': [
        {'C_0': (0.0, 10.0)}, {'C_h': (0.0, 1000), 'sigma_h': (0.1, 1.0)}]
}


def mean_squared_error(y_fit, y_data):
    return np.mean((y_fit - y_data)**2)


def main():
    phis = [0, 15, 30, 45]
    n_interp = 35
    #filepath = os.path.dirname(os.path.relpath(__file__))
    filepath = "/mnt/home/sshamloo/elecboltz/"

    name = "ADMR_NdLSCO_T25_relative_band=t+tp+tpp+tz_scat=hotspot" \
           "_free=C0+Ch+sigmah"
    path = pathlib.Path(filepath + name)
    path.mkdir(parents=True)

    loader = elecboltz.Loader(
        x_vary_label='theta', y_label='rho_zz',
        x_search={'phi': phis.copy(), 'H': [45] * 4},
        save_new_labels=False)
    loader.load(
        filepath + "data/ADMR_NdLSCO", f"NdLSCO_0p25_rho_c-vs-theta_25K_",
        x_columns=[0], y_columns=[1])
    loader.interpolate(n_interp, x_normalize=0)
    
    params = get_init_params()
    elecboltz.fit_model(
        loader.x_data, loader.y_data, init_params=params, bounds=bounds,
        x_normalize={'Btheta': 0}, loss=mean_squared_error,
        workers=-1, polish=False, save_path=path, save_label=name)


if __name__ == "__main__":
    main()