from .bandstructure import BandStructure
from .conductivity import Conductivity
from .params import easy_params

import numpy as np
from scipy.optimize import differential_evolution
import json
from datetime import datetime
from time import time
from copy import deepcopy
from multiprocessing import cpu_count
from pathlib import Path
from pprint import pformat
from typing import Callable
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
        ``easy_params`` to the ``BandStructure`` and ``Conductivity``.
    save_path : str, optional
        The directory where the fitting logs will be saved. If not
        provided, logs will not be saved.
    save_label : str, optional
        Label of the results.
    update_keys : Collection[str], optional
        The "flattened" keys of the parameters that will be updated
        during the fitting. See ``extract_keys`` for more information.
        Used for updating the ``params`` attribute and logging. If not
        provided, ``params`` will not be updated and the parameter names
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
    total_time : float
        The total time spent on the fitting routine.
    """
    def __init__(self, init_params: Mapping, save_path: str = None,
                 save_label: str = "fit", update_keys: Collection[str] = None,
                 print_log: bool = True):
        self.init_params = init_params
        self.save_path = save_path
        self.save_label = save_label
        self.update_keys = update_keys
        self.print_log = print_log
        self.iteration = 0
        self.last_time = time()
        self.total_time = 0.0

    def residual(
            self, param_values: Sequence, param_keys: Sequence[str],
            x_data: Mapping[str, Sequence], y_data: Mapping[str, Sequence],
            x_shift: Mapping = None, x_normalize: Mapping = None,
            y_shift: Mapping = None, y_normalize: Mapping = None,
            cond_obj: Conductivity = None, loss: Callable =
                lambda y_fit, y_data: np.mean(np.abs(y_fit - y_data))):
        """Compute the residual for the given parameters and data.

        Parameters
        ----------
        param_values : Sequence
            The values of the parameters to update.
        param_keys : Sequence[str]
            The keys of the parameters to update.
        x_data : Mapping[str, Sequence]
            The independent variable data (e.g. field). The name of the
            variable is mapped to the data, e.g.
            ``{'field': [0, 1, 2]}``.
        y_data : Mapping[str, Sequence]
            The dependent variable data (e.g. conductivity). The name
            of the variable is mapped to the data, e.g.
            ``{'sigma_xx': [1.1, 2.4, 3.8]}``.
        x_shift : Mapping, optional
            If provided, the y values will be shifted by the y value at
            this x point.
        x_normalize : Mapping, optional
            If provided, the y values will be normalized by the y value
            at this x point.  Note that shifts are applied before
            normalization.
        y_shift : float, optional
            The y values will be normalized to this value
            (if ``x_normalize`` is provided).
        y_normalize : float, optional
            The y values will be shifted to this value (if ``x_shift``
            is provided).
        squared : bool, optional
            If True, the residual is computed as the mean squared error.
            If False, it returns the mean absolute difference.
        cond_obj : Conductivity, optional
            If provided, the fitter will try to salvage the calculations
            already done from that object.
        """
        cond = self._build_obj(param_values, param_keys, cond_obj)
        name, y_label_i, y_label_j = self._get_label_indices(y_data.keys())

        y_fit = {label: np.zeros_like(y) for label, y in y_data.items()}
        for i in range(len(list(x_data.values())[0])):
            x = {label: x[i] for label, x in x_data.items()}
            y = _get_y(cond, x, y_data, name, y_label_i, y_label_j)
            for label in y_fit:
                y_fit[label][i] = y[label]
        if x_shift is not None:
            y0 = _get_y(cond, x_shift, y_data, name, y_label_i, y_label_j)
            for label in y_fit:
                y_fit[label] -= y0[label]
            if y_shift is not None:
                for label in y_fit:
                    y_fit[label] += y_shift[label]
        if x_normalize is not None:
            y0 = _get_y(cond, x_normalize, y_data, name, y_label_i, y_label_j)
            for label in y_fit:
                y_fit[label] /= y0[label]
            if y_normalize is not None:
                for label in y_fit:
                    y_fit[label] *= y_normalize[label]

        y_fit = np.concatenate(list(y_fit.values()))
        y_data = np.concatenate(list(y_data.values()))
        return loss(y_fit, y_data)

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
        self.total_time += iter_time

        log_message = f"Iteration {self.iteration}\n" \
                      f"----------{'-' * len(str(self.iteration))}\n" \
                      f"Best Parameters:\n"
        if self.update_keys is not None:
            update_params = _build_params_from_flat(
                self.update_keys, param_values)
            log_message += pformat(update_params) + "\n"
        else:
            log_message += pformat(param_values) + "\n"
        if convergence is not None:
            log_message += f"Convergence: {convergence:.5f}\n\n"
        log_message += f"Iteration Runtime: {iter_time:.3f} seconds\n"
        minutes, seconds = divmod(self.total_time, 60)
        hours, minutes = divmod(minutes, 60)
        log_message += f"Total Runtime: "
        if hours > 0:
            log_message += f"{int(hours)} hours "
        if minutes > 0:
            log_message += f"{int(minutes)} minutes "
        log_message += f"{seconds:.1f} seconds\n"

        if self.print_log:
            print(log_message)
        if self.save_path is not None:
            path = Path(self.save_path) / f"{self.save_label}.log"
            mode = 'w' if self.iteration == 1 else 'a'
            with open(path, mode) as log_file:
                log_file.write(log_message)

    def _build_obj(self, param_values: Sequence, param_keys: Sequence[str],
                   cond: Conductivity = None) -> Conductivity:
        """
        Build the conductivity object with the given parameters.
        """
        if cond is None:
            band = BandStructure(**easy_params(self.init_params))
            cond = Conductivity(band, **easy_params(self.init_params))
            update_band = True
        else:
            cond = deepcopy(cond)
            band = cond.band
            update_band = False
        params = deepcopy(self.init_params)
        for key, value in zip(param_keys, param_values):
            _update_flat_value(params, key, value)
        new_params = easy_params(params)
        for key, value in new_params.items():
            if key == 'band_params':
                for band_key, band_value in value.items():
                    if band.band_params[band_key] != band_value:
                        update_band = True
                        band.band_params[band_key] = band_value
            if hasattr(band, key):
                if np.any(getattr(band, key) != value):
                    update_band = True
                    setattr(band, key, value)
            if hasattr(cond, key):
                if np.any(getattr(cond, key) != value):
                    setattr(cond, key, value)
        if update_band:
            band.discretize()
            cond.band = band
        return cond

    def _get_label_indices(self, labels: Collection[str]):
        """Extract the names and indices from y_data keys."""
        name, i, j = {}, {}, {}
        for label in labels:
            name[label], index_labels = label.split("_")
            i[label] = {'x': 0, 'y': 1, 'z': 2}[index_labels[0]]
            j[label] = {'x': 0, 'y': 1, 'z': 2}[index_labels[1]]
        return name, i, j


def fit_model(x_data: Mapping[str, Sequence], y_data: Mapping[str, Sequence],
              init_params: Mapping, bounds: Mapping,
              x_shift: Mapping = None, x_normalize: Mapping = None,
              save_path: str = None, save_label: str = None,
              worker_percentage: float = 0.0, **kwargs):
    """Convenience function to set up and run a fitting routine.

    This uses ``scipy.optimize.differential_evolution`` to perform a
    global fit. The optimizer is hard-coded, because the callback
    functions are highly specific to each optimizer.
    Saves the results to the specified path.

    Parameters
    ----------
    x_data : Mapping[str, Sequence]
        The independent variable data (e.g. field). The name of the
        variable is mapped to the data, e.g. ``{'field': [0, 1, 2]}``.
    y_data : Mapping[str, Sequence]
        The dependent variable data (e.g. conductivity). The name of
        the variable is mapped to the data, e.g.
        ``{'sigma_xx': [1.1, 2.4, 3.8]}``. The name of each variable
        must start with "sigma" or "rho" (for conductivity or
        resistivity, respectively), and you can add a suffix to specify
        the component (e.g. ``"sigma_xx"``, ``"rho_xy"``).
    init_params : Mapping
        Initial parameters for the fitting routine, and also other
        parameters for initiallizing the classes. This is passed through
        ``easy_params`` to the ``BandStructure`` and ``Conductivity``.
    bounds : Mapping
        Bounds for the fitting parameters. This mapping has the same
        structure as ``init_params``, but only containing the variables
        that are to be fitted, and their values in the mapping must be
        a collection of the form (min, max).
    x_shift : Mapping, optional
        If provided, the y values will be shifted by the y value at
        this x point. The mapping must have the same structure as
        ``x_data``, but with single values instead of arrays as
        the values.
    x_normalize : Mapping, optional
        If provided, the y values will be normalized by the y value
        at this x point. Note that shifting will always be done
        before normalization. The mapping must have the same structure
        as ``x_data``, but with single values instead of arrays as
        the values. Note that shifts are applied before normalization.
    save_path : str, optional
        The directory where the fitting results will be saved.
        If not provided, results will not be saved.
    save_label : str, optional
        Label of the results. If not provided, will be set to
        ``f"y_label_x_label"``. If ``y_label` or ``x_label`` are
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

    update_keys = _extract_flat_keys(bounds, bounds=True)
    bounds = [_extract_flat_value(bounds, key) for key in update_keys]
    x0 = [_extract_flat_value(init_params, key) for key in update_keys]
    for i in range(len(update_keys)):
        if x0[i] is None:
            x_min, x_max = bounds[i]
            x0[i] = (x_min + x_max) / 2

    begin_time = datetime.now()
    cond = _prebuild_cond(init_params)
    fitter = FittingRoutine(init_params, save_path, save_label,
                            update_keys=update_keys)
    result = differential_evolution(
        fitter.residual, bounds=bounds, x0=x0, callback=fitter.log,
        args=(update_keys, x_data, y_data, x_shift, x_normalize, cond),
        **kwargs)
    end_time = datetime.now()

    print(result.message)
    return _save_fit_result(
        result, init_params, update_keys, begin_time, end_time,
        save_path, save_label)


def _prebuild_cond(init_params):
    band = BandStructure(**easy_params(init_params))
    band.discretize()
    cond = Conductivity(band, **easy_params(init_params))
    cond._build_elements()
    cond._build_differential_operator()
    return cond


def _extract_flat_keys(params, bounds=False):
    """Extract dots-separated keys from a nested structure.
    
    This function recursively extracts keys from a nested dictionary
    containing dictionaries and lists, and returns a flat list of keys.
    Each key is represented as a string with dot-separated keys and
    indices, e.g. ``"band_params.a"``, is ``params["band_params"]["a"]``
    or ``"scattering_params.nu.0"`` is
    ``params["scattering_params"]["nu"][0]``.

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
            value = params[key]
            if _is_value_nested(value, bounds):
                for val_key in _extract_flat_keys(value, bounds=bounds):
                    keys.append(f"{key}.{val_key}")
            else:
                keys.append(key)
    elif isinstance(params, Sequence):
        for (i, value) in enumerate(params):
            if _is_value_nested(value, bounds):
                for val_key in _extract_flat_keys(value, bounds=bounds):
                    keys.append(f"{i}.{val_key}")
            else:
                keys.append(str(i))
    return keys


def _is_value_nested(value, bounds):
    if bounds:
        if isinstance(value, Mapping):
            return any(isinstance(v, Collection) and
                       not isinstance(v, str) for v in value.values())
        elif isinstance(value, Collection):
            return any(isinstance(v, Collection) and
                       not isinstance(v, str) for v in value)
    else:
        return isinstance(value, Collection) and not isinstance(value, str)


def _extract_flat_value(params: Mapping, flat_key: str):
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


def _update_flat_value(params: Mapping, flat_key: str, value):
    level_params = params
    key_parts = flat_key.split('.')
    while len(key_parts) > 1:
        key = key_parts.pop(0)
        if str.isnumeric(key):
            key = int(key)
            if key >= len(level_params):
                return
        elif key not in level_params:
            return
        level_params = level_params[key]
    key = key_parts[0]
    if str.isnumeric(key):
        key = int(key)
        if key >= len(level_params):
            return
    level_params[key] = value


def _build_params_from_flat(param_keys, param_values):
    """Build a nested dictionary from flattened keys and values.

    Parameters
    ----------
    param_keys : Collection[str]
        The "flattened" keys of the parameters. See ``extract_keys``
        for more information.
    param_values : Sequence
        The values corresponding to the keys in ``param_keys``.

    Returns
    -------
    dict
        A nested dictionary where the keys are the flattened keys
        and the values are the corresponding values from ``param_values``.
    """
    params = dict()
    param_values = list(param_values)
    for key in param_keys:
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
        level_params[key] = param_values.pop(0)
    return params


def _get_y(cond, x_data, y_data, name, y_label_i, y_label_j):
    y = {}
    for label, x in x_data.items():
        setattr(cond, label, x)
    if 'rho' in name.values():
        cond.calculate()
        rho = np.linalg.inv(cond.sigma)
    else:
        cond.calculate(sorted(set(y_label_i.values())),
                       sorted(set(y_label_j.values())))
    for label in y_data:
        if name[label] == 'sigma':
            y[label] = cond.sigma[y_label_i[label], y_label_j[label]]
        elif name[label] == 'rho':
            y[label] = rho[y_label_i[label], y_label_j[label]]
        else:
            raise ValueError(f"Unknown y_data key: {name[label]}")
    return y


def _save_fit_result(result, init_params, update_keys, begin_time,
                     end_time, save_path, save_label):
    result = _result_to_serializable(result)
    result['fit_params'] = _build_params_from_flat(update_keys, result['x'])
    result['residual'] = result['fun']
    result['evaluations'] = result['nfev']
    result['iterations'] = result['nit']
    result.pop('x')
    result.pop('population')
    result.pop('population_energies')
    result.pop('fun')
    result.pop('evalulations')

    result['init_params'] = _build_params_from_flat(
        update_keys, [_extract_flat_value(init_params, key)
                      for key in update_keys])
    all_keys = _extract_flat_keys(init_params, bounds=False)
    fixed_keys = set(all_keys) - set(update_keys)
    result['fixed_params'] = _build_params_from_flat(
        fixed_keys, [_extract_flat_value(init_params, key)
                     for key in fixed_keys])

    result['begin_time'] = begin_time.isoformat()
    result['end_time'] = end_time.isoformat()
    result['runtime'] = (end_time - begin_time).total_seconds()

    if save_path is not None:
        path = Path(save_path) / f"{save_label}.json"
        with path.open('w') as f:
            json.dump(result, f, indent=2)
    return result


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
