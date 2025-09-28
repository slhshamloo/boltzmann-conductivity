from .bandstructure import BandStructure
from .conductivity import Conductivity
from .params import easy_params

import numpy as np
from scipy.optimize import differential_evolution
import json
from datetime import datetime
from time import time
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Union, Callable
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
            x_data: Mapping[str, Union[Sequence, Sequence[Sequence]]],
            y_data: Mapping[str, Union[Sequence, Sequence[Sequence]]],
            multi_params: Collection = [], x_shift: Mapping = None,
            x_normalize: Mapping = None, y_shift: Mapping = None,
            y_normalize: Mapping = None, cond_obj: Conductivity = None,
            loss: Callable = lambda y_fit, y_data: np.mean(
                np.abs(y_fit - y_data)),
            postprocess: Callable = lambda x, y: y):
        """Compute the residual for the given parameters and data.

        Parameters
        ----------
        param_values : Sequence
            The values of the parameters to update.
        param_keys : Sequence[str]
            The "flat keys of the parameters to update.
        x_data : Mapping[str, Union[Sequence, Sequence[Sequence]]]
            The independent variable data (e.g. field). The name of the
            variable is mapped to the data, e.g.
            ``{'field': [0.5, 1.5, 2.5]}``. In case of a nonempty
            ``multi_params``, the value must be a collection of
            sequences, where each sequence corresponds to a different
            parameter to be fitted,
            e.g. ``{'field': [[0.5, 1.5, 2.5], [0.6, 1.6, 2.6]]}``.
        y_data : Mapping[str, Union[Sequence, Sequence[Sequence]]]
            The dependent variable data (e.g. conductivity). The name of
            the variable is mapped to the data, e.g.
            ``{'sigma_xx': [1.1, 2.4, 3.8]}``. The name of each variable
            must start with "sigma" or "rho" (for conductivity or
            resistivity, respectively), and you can add a suffix to
            specify the component (e.g. ``"sigma_xx"``, ``"rho_xy"``).
            In case of nonempty ``multi_params``, the value must be a
            collection of sequences, where each sequence corresponds to
            a different parameter to be fitted, e.g.
            ``{'rho_zz': [[1.1, 2.4, 3.8], [1.2, 2.5, 3.9]]}``.
        multi_params : Collection, optional
            A collection of parameters that are to be fitted differently
            for the different datasets in ``x_data`` and ``y_data``, if
            there is more than one. To make it precise, each label must
            be a dot-separated string, showing the "path" to the value
            in the parameters dictionary, e.g. ``"band_params.mu"`` or
            ``"scattering_params.nu.0"``. These parameters must
            themselves be collections in the parameters dictionary,
            showing the value for every dataset e.g. ``{'band_params':
            {'mu': [0.1, 0.2, 0.3]}}`` in ``init_params`` or
            ``{'band_params': {'mu': [(0.1, 0.9), (0.2, 0.8),
            (0.3, 0.7)]}}`` in ``bounds``.
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
        cond_obj : Conductivity, optional
            If provided, the fitter will try to salvage the
            calculations already done from that object.
        loss : Callable, optional
            A function that takes the fit and data y values, and
            returns a scalar loss value. By default, the mean absolute
            error is used.
        postprocess : Callable, optional
            This callable is applied to the data and fit y values
            before calculating the loss. It takes ``x_data`` and
            a ``y`` with a format similar to ``y_data``, and returns
            the processed ``y``, again with a format similar to
            ``y_data``. By default, no postprocessing is applied. An
            example use case is filtering out parts of the values
            where the data can be unreliable.
        """
        if multi_params:
            return self._calculate_multi(
                param_values, param_keys, x_data, y_data, multi_params,
                x_shift, x_normalize, y_shift, y_normalize, cond_obj,
                loss, postprocess)
        else:
            return self._calculate_single(
                param_values, param_keys, x_data, y_data,
                x_shift, x_normalize, y_shift, y_normalize, cond_obj,
                loss, postprocess)

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

    def _calculate_single(self, param_values, param_keys, x_data, y_data,
                          x_shift, x_normalize, y_shift, y_normalize,
                          cond_obj, loss, postprocess):
        params = deepcopy(self.init_params)
        for key, value in zip(param_keys, param_values):
            _update_flat_value(params, key, value)
        params = easy_params(params)
        cond = self._build_obj(params, cond_obj)

        name, y_label_i, y_label_j = self._get_label_indices(y_data.keys())
        y_fit = {label: np.zeros_like(y) for label, y in y_data.items()}
        for i in range(len(next(iter(x_data.values())))):
            x = {label: x[i] for label, x in x_data.items()}
            y = _calc_y(cond, x, y_data, name, y_label_i, y_label_j)
            for label in y_fit:
                y_fit[label][i] = y[label]
        if x_shift is not None:
            y0 = _calc_y(cond, x_shift, y_data, name, y_label_i, y_label_j)
            for label in y_fit:
                y_fit[label] -= y0[label]
            if y_shift is not None:
                for label in y_fit:
                    y_fit[label] += y_shift[label]
        if x_normalize is not None:
            y0 = _calc_y(cond, x_normalize, y_data, name, y_label_i, y_label_j)
            for label in y_fit:
                y_fit[label] /= y0[label]
            if y_normalize is not None:
                for label in y_fit:
                    y_fit[label] *= y_normalize[label]
        y_fit = postprocess(x_data, y_fit)
        y_data = postprocess(x_data, y_data)
        y_fit = np.concatenate(list(y_fit.values()))
        y_data = np.concatenate(list(y_data.values()))
        return loss(y_fit, y_data)
    
    def _calculate_multi(self, param_values, param_keys, x_data, y_data,
                         multi_params, x_shift, x_normalize,
                         y_shift, y_normalize, cond_obj, loss, postprocess):
        n_data_sets = len(next(iter(x_data.values())))
        total_loss = 0.0
        name, y_label_i, y_label_j = self._get_label_indices(y_data.keys())
        params = deepcopy(self.init_params)
        for key, value in zip(param_keys, param_values):
            _update_flat_value(params, key, value)
        params_data_set = deepcopy(params)
        for i in range(n_data_sets):
            y_fit = {label: np.zeros_like(y[i]) for label, y in y_data.items()}
            for multi_param in multi_params:
                _update_flat_value(
                    params_data_set, multi_param,
                    _extract_flat_value(params, multi_param)[i])
            params_data_set = easy_params(params_data_set)
            cond = self._build_obj(params_data_set, cond_obj)
            for j in range(len(next(iter(y_fit.values())))):
                x = {label: x[i][j] for label, x in x_data.items()}
                y = _calc_y(cond, x, y_data, name, y_label_i, y_label_j)
                for label in y_fit:
                    y_fit[label][j] = y[label]
            if x_shift is not None:
                x = {label: x_shift[label][i] for label in x_shift}
                y0 = _calc_y(cond, x, y_data, name, y_label_i, y_label_j)
                for label in y_fit:
                    y_fit[label] -= y0[label]
                if y_shift is not None:
                    for label in y_fit:
                        y_fit[label] += y_shift[label][i]
            if x_normalize is not None:
                x = {label: x_normalize[label][i] for label in x_normalize}
                y0 = _calc_y(cond, x, y_data, name, y_label_i, y_label_j)
                for label in y_fit:
                    y_fit[label] /= y0[label]
                if y_normalize is not None:
                    for label in y_fit:
                        y_fit[label] *= y_normalize[label][i]
            x_data_set = {label: x[i] for label, x in x_data.items()}
            y_fit = postprocess(x_data_set, y_fit)
            y_data_set = postprocess(x_data_set, y_data_set)
            total_loss += loss(y_fit, y_data_set) / n_data_sets
        return total_loss

    def _build_obj(self, params, cond):
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
        for key, value in params.items():
            if key == 'band_params':
                for band_key, band_value in value.items():
                    if band.band_params[band_key] != band_value:
                        update_band = True
                        band.band_params[band_key] = band_value
            elif key == 'scattering_params':
                for cond_key, cond_value in value.items():
                    if hasattr(cond, cond_key):
                        if np.any(getattr(cond, cond_key) != cond_value):
                            setattr(cond, cond_key, cond_value)
            elif hasattr(band, key):
                if np.any(getattr(band, key) != value):
                    update_band = True
                    setattr(band, key, value)
            elif hasattr(cond, key):
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


def _mean_absolute_error(y_fit, y_data):
    """Mean absolute error loss function."""
    return np.mean(np.abs(y_fit - y_data))


def _dummy_postprocess(x, y):
    """A dummy postprocessing function that does nothing.
    Exists because lambdas cannot be pickled, and so cannot be used in
    multiprocessed fitting routines."""
    return y


def fit_model(x_data: Mapping[str, Union[Sequence, Sequence[Sequence]]],
              y_data: Mapping[str, Union[Sequence, Sequence[Sequence]]],
              init_params: Mapping, bounds: Mapping,
              multi_params: Collection[str] = [],
              x_shift: Mapping = None, x_normalize: Mapping = None,
              y_shift: Mapping = None, y_normalize: Mapping = None,
              loss: Callable = _mean_absolute_error,
              postprocess: Callable = _dummy_postprocess,
              save_path: str = None, save_label: str = None, **kwargs):
    """Convenience function to set up and run a fitting routine.

    This uses ``scipy.optimize.differential_evolution`` to perform a
    global fit. The optimizer is hard-coded, because the callback
    functions are highly specific to each optimizer.
    Saves the results to the specified path.

    Parameters
    ----------
    x_data : Mapping[str, Union[Sequence, Sequence[Sequence]]]
        The independent variable data (e.g. field). The name of the
        variable is mapped to the data, e.g.
        ``{'field': [[0.5, 1.5, 2.5], [0.6, 1.6, 2.6]]}``.
        In case of nonempty ``multi_params``, the value must be a
        collection of sequences, where each sequence corresponds to a
        different parameter to be fitted,
        e.g. ``{'field': [[0, 1, 2], [0, 1, 2]]}``.
    y_data : Mapping[str, Union[Sequence, Sequence[Sequence]]]
        The dependent variable data (e.g. conductivity). The name of
        the variable is mapped to the data, e.g.
        ``{'sigma_xx': [1.1, 2.4, 3.8]}``. The name of each variable
        must start with "sigma" or "rho" (for conductivity or
        resistivity, respectively), and you can add a suffix to specify
        the component (e.g. ``"sigma_xx"``, ``"rho_xy"``). In case of
        nonempty ``multi_params``, the value must be a collection of
        sequences, where each sequence corresponds to a different
        parameter to be fitted, e.g. ``{'rho_zz': [[1.1, 2.4, 3.8],
        [1.2, 2.5, 3.9]]}``.
    init_params : Mapping
        Initial parameters for the fitting routine, and also other
        parameters for initiallizing the classes. This is passed through
        ``easy_params`` to the ``BandStructure`` and ``Conductivity``.
    bounds : Mapping
        Bounds for the fitting parameters. This mapping has the same
        structure as ``init_params``, but only containing the variables
        that are to be fitted, and their values in the mapping must be
        a collection of the form (min, max).
    multi_params : Collection, optional
        A collection of parameters that are to be fitted differently for
        the different datasets in ``x_data`` and ``y_data``, if there is
        more than one. To make it precise, each label must be a
        dot-separated string, showing the "path" to the value in the
        parameters dictionary, e.g. ``"band_params.mu"`` or
        ``"scattering_params.nu.0"``. These parameters must themselves
        be collections in the parameters dictionary, showing the value
        for every dataset e.g.
        ``{'band_params': {'mu': [0.1, 0.2, 0.3]}}`` in ``init_params``
        or ``{'band_params': {'mu': [(0.1, 0.9), (0.2, 0.8),
        (0.3, 0.7)]}}`` in ``bounds``.
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
    y_shift : Mapping, optional
        The y values will be normalized to this value (if
        ``x_normalize`` is provided). The mapping must have the same
        structure as ``y_data``, but with single values instead of
        arrays as the values.
    y_normalize : Mapping, optional
        The y values will be shifted to this value (if ``x_shift`` is
        provided). The mapping must have the same structure as
        ``y_data``, but with single values instead of arrays as the
        values.
    loss : Callable, optional
        A function that takes the fit and data y values, and returns a
        scalar loss value. By default, the mean absolute error is used.
    postprocess : Callable, optional
        This callable is applied to the data and fit y values before
        calculating the loss. It takes ``x_data`` and a ``y`` with a
        format similar to ``y_data``, and returns the processed ``y``,
        again with a format similar to ``y_data``. By default, no
        postprocessing is applied. An example use case is filtering out
        parts of the values where the data can be unreliable.
    save_path : str, optional
        The directory where the fitting results will be saved.
        If not provided, results will not be saved.
    save_label : str, optional
        Label of the results. If not provided, will be set to
        ``f"y_label_x_label"``. If ``y_label` or ``x_label`` are
        collections of string, they will be joined with an underscore.
    **kwargs : dict, optional
        Additional keyword arguments passed to
        `scipy.optimize.differential_evolution`.
    """
    if save_label is None:
        x_string = x_label if isinstance(x_label, str) else "_".join(x_label)
        y_string = y_label if isinstance(y_label, str) else "_".join(y_label)
        save_label = f"{y_string}_{x_string}"
    
    update_keys = _extract_flat_keys(bounds, bounds=True)
    bounds = [_extract_flat_value(bounds, key) for key in update_keys]
    x0 = [_extract_flat_value(init_params, key) for key in update_keys]
    if multi_params:
        update_keys, bounds, x0, init_params = _multiply_multi_params(
            update_keys, bounds, x0, init_params,
            multi_params, len(next(iter(x_data.values()))))
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
        args=(update_keys, x_data, y_data, multi_params,
              x_shift, x_normalize, y_shift, y_normalize, cond,
              loss, postprocess),
        **kwargs)
    end_time = datetime.now()

    print(result.message)
    return _save_fit_result(
        result, init_params, update_keys, begin_time, end_time,
        save_path, save_label)


def _multiply_multi_params(update_keys, bounds, x0, init_params,
                           multi_params, n):
    """Expand multi-parameters in the update keys, bounds, and x0 lists."""
    i = 0
    while i < len(update_keys):
        if update_keys[i] in multi_params: # multi-parameter with single bound
            if not isinstance(x0[i], Sequence):
                init_param_list = []
            for j in range(n):
                update_keys.insert(i + j + 1, f"{update_keys[i]}.{j}")
                bounds.insert(i + j + 1, bounds[i])
                if isinstance(x0[i], Sequence):
                    x0.insert(i + j + 1, x0[i][j])
                else:
                    x0.insert(i + j + 1, x0[i])
                    init_param_list.append(x0[i])
            if not isinstance(x0[i], Sequence):
                _update_flat_value(init_params, update_keys[i],
                                   init_param_list)
            update_keys.pop(i)
            bounds.pop(i)
            x0.pop(i)
            i += n
        else:
            parent = '.'.join(update_keys[i].split('.')[:-1])
            if parent in multi_params and x0[i] is None:
                parent_value = _extract_flat_value(init_params, parent)
                if isinstance(parent_value, Sequence):
                    _update_flat_value(init_params, parent,
                                       parent_value + [parent_value[0]])
                    x0[i] = parent_value[0]
                elif parent_value is not None:
                    _update_flat_value(init_params, parent, [parent_value])
                    x0[i] = parent_value
            i += 1
    return update_keys, bounds, x0, init_params


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
        if str.isnumeric(key) and not isinstance(value, Mapping):
            key = int(key)
            if isinstance(value, Sequence):
                if key >= len(value):
                    return None
            else:
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


def _calc_y(cond, x_data, y_data, name, y_label_i, y_label_j):
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
    result.pop('nfev')
    result.pop('nit')

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
