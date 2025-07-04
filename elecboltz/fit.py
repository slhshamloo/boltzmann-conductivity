from .bandstructure import BandStructure
from .conductivity import Conductivity
from .params import easy_params

import numpy as np
import scipy.optimize
import json
from copy import copy, deepcopy
from time import time
from multiprocessing import cpu_count
from pathlib import Path
from pprint import pformat

from typing import Any, Union, Callable
from collections.abc import Sequence, Collection, Mapping


class FittingRoutine:
    """This convenience class automatically sets up a fitting routine.

    Stores the necessary data and generates the functions used in
    This includes handling the residual computation, parameter update,
    logging the fitting progress, and saving the results to a file.

    Parameters
    ----------
    init_params : Mapping
        Initial parameters for the fitting routine, and also other
        parameters for initiallizing the classes. This is passed through
        `easy_params` to the `BandStructure` and `Conductivity`.
    save_path : str, optional
        The directory where the fitting logs will be saved. If not
        provided, logs will not be saved.
    save_label : str, optional
        Label of the results.
    update_keys : Collection[str], optional
        The "flattened" keys of the parameters that will be updated
        during the fitting. See `extract_keys` for more information.
        Used for updating the `params` attribute and logging. If not
        provided, `params` will not be updated and the parameter names
        will not be mentioned in the log.
    print_log : bool, optional
        If True, the fitting progress will be printed to the console.
    
    Attributes
    ----------
    params : Mapping
        The current parameters dictionary.
    save_path : Path
        The directory where the fitting logs will be saved.
    save_label : Path
        The label of the results.
    update_keys : Collection[str]
        The "flattened" keys of the parameters that will be updated
        during the fitting. This is used for updating the `params`
        attribute and logging.
    iteration : int
        The current iteration number.
    last_time : float
        The timestamp of the last fitting iteration.
    base_band : BandStructure
        The base band structure object.
    base_cond : Conductivity
        The base conductivity object.

    """
    def __init__(self, init_params: Mapping, save_path: str = None,
                 save_label: str = "fit", update_keys: Collection[str] = None,
                 print_log: bool = True):
        self.params = init_params
        self.save_path = save_path
        self.save_label = save_label
        self.update_keys = update_keys
        self.print_log = print_log
        self.iteration = 0
        self.last_time = time()
        self.base_band = BandStructure(**easy_params(init_params))
        self.base_band.discretize()
        self.base_cond = Conductivity(
            self.base_band, **easy_params(init_params))
        self.base_cond._build_elements()
        self.base_cond._build_differential_operator()

    def residual(self, param_values: Sequence, param_keys: Sequence[str],
                 x_data: Union[np.ndarray, Sequence[np.ndarray]],
                 y_data: Union[np.ndarray, Sequence[np.ndarray]],
                 x_label: Union[str, Sequence[str]],
                 y_label: Union[str, Sequence[str]],
                 squared: bool = True):
        """Compute the residual for the given parameters and data.

        Parameters
        ----------
        param_values : Sequence
            The values of the parameters to update.
        param_keys : Sequence[str]
            The keys of the parameters to update.
        x_data : Union[numpy.ndarray, Sequence[numpy.ndarray]]
        The independent variable data (e.g. field). If multiple datasets
        are provided, they should be in a collection of numpy arrays.
        y_data : numpy.ndarray
            The dependent variable data (e.g. conductivity). If multiple
            datasets are provided, they should be in a collection of
            numpy arrays.
        x_label : Union[str, Sequence[str]]
            The label(s) for the independent variable(s).
        y_label : Union[numpy.ndarray, Sequence[numpy.ndarray]]
            The label(s) for the dependent variable(s). Each label can
            start with "sigma" or "rho" (for conductivity or resistivity,
            respectively), and you can add a suffix to specify the
            component (e.g. "sigma_xx", "rho_xy").
        squared : bool, optional
            If True, the residual is computed as the mean squared error.
            If False, it returns the mean absolute difference.
        """
        cond = self._build_obj(param_values, param_keys)
        x_data, y_data, x_label, y_label = \
            _pack_single_values(x_data, y_data, x_label, y_label)
        name, y_label_i, y_label_j = self._get_calc_indices(y_label)

        y_fit = [np.zeros_like(y) for y in y_data]
        for i in range(len(x_data[0])):
            for label, x in zip(x_label, x_data):
                setattr(cond, label, x[i])
            cond.calculate(sorted(set(y_label_i)), sorted(set(y_label_j)))
            if 'rho' in name:
                rho = np.linalg.inv(cond.sigma)
            for j, label in enumerate(y_label):
                if name[j] == 'sigma':
                    y_fit[j][i] = cond.sigma[y_label_i[j], y_label_j[j]]
                elif name[j] == 'rho':
                    y_fit[j][i] = rho[y_label_i[j], y_label_j[j]]
                else:
                    raise ValueError(f"Unknown y_label: {name[j]}")

        y_fit = np.concatenate(y_fit)
        y_data = np.concatenate(y_data)
        if squared:
            return np.mean(np.abs(y_fit-y_data) ** 2) # abs for complex data
        else:
            return np.mean(np.abs(y_fit-y_data))

    def log(self, param_values, convergence: float = None):
        """Log the current fitting iteration and parameters.
        
        Parameters
        ----------
        param_values : Sequence
            The current values of the parameters.
        convergence : float, optional
            The convergence value of the fitting routine. 0.0 means
            no convergence, 1.0 means perfect convergence.
            If not provided, it will not be logged.
        """
        self.iteration += 1
        now = time()
        iter_time = now - self.last_time
        self.last_time = now

        log_message = f"Iteration {self.iteration}\n" \
                      f"----------{'-' * len(str(self.iteration))}\n" \
                      f"Best Parameters:\n"
        if self.update_keys is not None:
            update_params = _build_params_from_flat(
                self.update_keys, param_values)
            self.params.update(update_params)
            log_message += pformat(update_params) + "\n"
        else:
            log_message += pformat(param_values) + "\n"
        if convergence is not None:
            log_message += f"Convergence: {convergence:.5f}\n\n"
        log_message += f"Iteration Runtime: {iter_time:.3f} seconds\n"
        
        if self.print_log:
            print(log_message)
        if self.save_path is not None:
            path = Path(self.save_path) / f"{self.save_label}.log"
            with open(path, 'a') as log_file:
                log_file.write(log_message)

    def _build_obj(self, param_values: Sequence,
                   param_keys: Sequence[str]):
        """
        Build the conductivity object with the given parameters.
        """
        BandStructure.__init__.__code__.co_varnames[1:-1]
        band = copy(self.base_band)
        cond = deepcopy(self.base_cond)
        params = _build_params_from_flat(param_values, param_keys)
        new_params = easy_params(params)
        update_band = False
        for key, value in new_params.items():
            if hasattr(band, key):
                update_band = True
                setattr(band, key, value)
            if hasattr(cond, key):
                setattr(cond, key, value)
        if update_band:
            band.discretize()
            cond.band = band
        return cond

    def _get_label_indices(self, labels: Sequence[str]):
        """Extract the names and indices from y_label."""
        name, i, j = [], [], []
        for label in labels:
            name_label, index_labels = label.split("_")
            name.append(name_label)
            i.append({'x': 0, 'y': 1, 'z': 2}[index_labels[0]])
            j.append({'x': 0, 'y': 1, 'z': 2}[index_labels[1]])
        return name, i, j


def fit_model(x_data: Union[np.ndarray, Sequence[np.ndarray]],
              y_data: Union[np.ndarray, Sequence[np.ndarray]],
              x_label: Union[str, Sequence[str]],
              y_label: Union[str, Sequence[str]],
              init_params: Mapping, bounds: Mapping, save_path: str = None,
              save_label: str = None, worker_percentage: float = 0.0,
              **kwargs):
    """Convenience function to set up and run a fitting routine.

    This uses `scipy.optimize.differential_evolution` to perform a
    global fit. The optimizer is hard-coded, because the callback
    functions are highly specific to each optimizer.
    Saves the results to the specified path.

    Parameters
    ----------
    x_data : Union[numpy.ndarray, Sequence[numpy.ndarray]]
        The independent variable data (e.g. field). If multiple datasets
        are provided, they should be in a collection of numpy arrays.
    y_data : numpy.ndarray
        The dependent variable data (e.g. conductivity). If multiple
        datasets are provided, they should be in a collection of
        numpy arrays.
    x_label : Union[str, Sequence[str]]
        The label(s) for the independent variable(s).
    y_label : Union[numpy.ndarray, Sequence[numpy.ndarray]]
        The label(s) for the dependent variable(s). Each label can
        start with "sigma" or "rho" (for conductivity or resistivity,
        respectively), and you can add a suffix to specify the
        component (e.g. "sigma_xx", "rho_xy").
    init_params : Mapping
        Initial parameters for the fitting routine, and also other
        parameters for initiallizing the classes. This is passed through
        `easy_params` to the `BandStructure` and `Conductivity`.
    bounds : Mapping
        Bounds for the fitting parameters. This mapping has the same
        structure as `init_params`, but only containing the variables
        that are to be fitted, and their values in the mapping must be
        a collection of the form (min, max).
    save_path : str, optional
        The directory where the fitting results will be saved.
        If not provided, results will not be saved.
    save_label : str, optional
        Label of the results. If not provided, will be set to
        ``f"y_label_x_label"``. If `y_label` or `x_label` are
        collections of string, they will be joined with an underscore.
    worker_percentage : float, optional
        The percentage of available workers to use for parallel
        computation. If set to 0, it will not be used for setting the
        number of workers. The number of workers can also be set by
        the `workers` keyword argument of differential evolution.
    log_format : str, optional
        The format for logging parameter values.
    **kwargs : dict, optional
        Additional keyword arguments passed to
        `scipy.optimize.differential_evolution`.
    """
    if save_label is None:
        x_string = x_label if isinstance(x_label, str) else "_".join(x_label)
        y_string = y_label if isinstance(y_label, str) else "_".join(y_label)
        save_label = f"{y_string}_{x_string}"
    if worker_percentage > 0:
        if "workers" in kwargs:
            raise ValueError(
                "Cannot set both `worker_percentage` and `workers`.")
        kwargs["workers"] = int(np.ceil(worker_percentage / 100 * cpu_count()))

    update_keys = _extract_flat_keys(bounds)
    bounds = [_extract_flat_value(bounds, key) for key in update_keys]
    x0 = [_extract_flat_value(init_params, key) for key in update_keys]
    for i in range(len(update_keys)):
        if x0[i] is None:
            x_min, x_max = bounds[i]
            x0[i] = (x_min + x_max) / 2

    fitter = FittingRoutine(init_params, save_path, save_label,
                            update_keys=update_keys)
    result = scipy.optimize.differential_evolution(
        fitter.residual, bounds=bounds, x0=x0,
        args=(update_keys, x_data, y_data, x_label, y_label), **kwargs)
    
    result = _result_to_serializable(result)
    result['fit_params'] = _build_params_from_flat(update_keys, result['x'])
    result.pop('x')
    all_keys = _extract_flat_keys(init_params)
    fixed_keys = set(all_keys) - set(update_keys)
    result['fixed_params'] = _build_params_from_flat(
        fixed_keys, [_extract_flat_value(init_params, key)
                     for key in fixed_keys])
    if save_path is not None:
        path = Path(save_path) / f"{save_label}.json"
        with path.open('w') as f:
            json.dump(result, f)
    return result


def _extract_flat_keys(params: Mapping) -> list[str]:
    """Extract dots-separated keys from a nested structure.
    
    This function recursively extracts keys from a nested dictionary
    containing dictionaries and lists, and returns a flat list of keys.
    Each key is represented as a string with dot-separated keys and
    indices, e.g. "band_params.a", is params["band_params"]["a"]
    or "scattering_params.nu.0" is params["scattering_params"]["nu"][0].

    Parameters
    ----------
    params : Mapping
        The nested dictionary from which to extract keys.
    
    Returns
    -------
    list[str]
        A list of flattened keys, where each key is a string
        representing the path to the value in the nested structure.
    """
    keys = []
    if isinstance(params, Mapping):
        for key in params:
            val = params[key]
            if any(isinstance(v, Collection) for v in val):
                for val_key in _extract_flat_keys(val):
                    keys.append(f"{key}.{val_key}")
            else:
                keys.append(key)
    elif isinstance(params, Sequence):
        for (i, val) in enumerate(params):
            if any(isinstance(v, Collection) for v in val):
                for val_key in _extract_flat_keys(val):
                    keys.append(f"{i}.{val_key}")
            else:
                keys.append(str(i))
    return keys


def _extract_flat_value(params: Mapping, flat_key: str) -> Any:
    """Extract value from a nested dictionary using a flattened key.

    Parameters
    ----------
    params : Mapping
        The nested dictionary from which to extract values.
    flat_key : str
        The "flattened" key of the parameter. see `extract_flat_keys`
        for more information.

    Returns
    -------
    Any
        The value corresponding to the flattened key in the nested
        dictionary, or None if the key does not exist.
    """
    value = params
    key_parts = flat_key.split('.')
    while key_parts:
        key = key_parts.pop(0)
        if str.isnumeric(key):
            key = int(key)
            if key >= len(value):
                return None
        elif key not in value:
            return None
        value = value[key]
    return value


def _build_params_from_flat(params_keys, params_values):
    """Build a nested dictionary from flattened keys and values.

    Parameters
    ----------
    params_keys : Collection[str]
        The "flattened" keys of the parameters. See `extract_keys`
        for more information.
    params_values : Sequence
        The values corresponding to the keys in `params_keys`.

    Returns
    -------
    dict
        A nested dictionary where the keys are the flattened keys
        and the values are the corresponding values from `params_values`.
    """
    params = dict()
    for key in params_keys:
        key_parts = key.split('.')
        key = key_parts[0]
        level_params = params
        for part in key_parts[1:]:
            if str.isnumeric(part):
                part = int(part)
                if isinstance(level_params, dict):
                    if key not in level_params:
                        level_params[key] = []
                if isinstance(level_params, list):
                    if level_params[key] == dict():
                        level_params[key] = []
                while part >= len(level_params[key]):
                    level_params[key].append(dict())
            elif key not in level_params:
                level_params[key] = dict()
            level_params = level_params[key]
            key = part
        level_params[key] = params_values.pop(0)
    return params
            

def _pack_single_values(x_data, y_data, x_label, y_label):
    """Ensure that single values are packed into lists for consistency."""
    if isinstance(x_data, np.ndarray):
        x_data = [x_data]
    if isinstance(y_data, np.ndarray):
        y_data = [y_data]
    if isinstance(x_label, str):
        x_label = [x_label]
    if isinstance(y_label, str):
        y_label = [y_label]
    return x_data, y_data, x_label, y_label


def _result_to_serializable(result):
    serializable = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, np.floating):
            serializable[key] = float(value)
        elif isinstance(value, np.integer):
            serializable[key] = int(value)
        else:
            try:
                json.dumps(value)  # test serializability
                serializable[key] = value
            except (TypeError, OverflowError):
                serializable[key] = str(value)
    return serializable
