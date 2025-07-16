import numpy as np
import re
from typing import Mapping, Sequence, Union
from pathlib import Path
from collections import defaultdict


class Loader:
    """Convenience class for loading data split into multiple files.
    
    Prepares the data for ``fit_model`` by processing multiple files
    labeled with values of independent variables and containing the
    variation of one variable. For example, the files can be labeled
    with the values of ``phi`` and ``field``, and contain the variation
    of ``theta``.

    Parameters
    ----------
    x_vary_label : Union[str, Sequence[str]], optional
        Label of the independent variable(s) that varies inside the
        files. If not provided, it will be inferred from the file
        contents (column headers).
    x_search : Mapping[str, Sequence[Union[int, float]]], optional
        Dictionary mapping labels of independent variables to the values
        to be searched for in the file names. Example: 
        ``{'phi': [30, 60], 'field': [1.3, 2.7]}``.
    y_label : Sequence[str], optional
        Labels of the dependent variables, used for fitting routine.
        See ``elecboltz.fit.fit_model`` for details about the allowed
        labels. If None, will be inferred from the file contents
        (column headers).
    save_new_labels : bool, optional
        If True, new labels found in the file names will be saved to
        ``x_search``.
    save_new_values : bool, optional
        If True, new values for existing labels in ``x_search`` will be
        added to the list of values for that label.
    data_type : {'plain', 'admr'}, optional
        Type of data to load. If ``'plain'``, no processing is done on
        the data and the x_data is constructed directly from the
        collected values. If ``'admr'``, the x_data is constructed
        such that it contains the field vector at each index.
    
    Attributes
    ----------
    x_data : defaultdict[str, list[np.ndarray]]
        x_data variable for the fitting procedure.
    y_data : defaultdict[str, list[np.ndarray]]
        y_data variable for the fitting procedure.
    x_data_raw : defaultdict[str, list[np.ndarray]]
        Raw data of the independent variable(s) collected from the
        files. Each ``x_vary_label`` is mapped to the corresponding
        arrays. Each array corresponds to a different value of the
        labels in ``x_search``.
    y_data_raw : deefaultdict[str, list[np.ndarray]]
        Raw data of the dependent variable collected from the files.
        Each sequence corresponds to a different dependent variable
        in ``y_label``, and each array inside that corresponds
        to a different value in ``x_search_values``.
    x_data_interpolated : defaultdict[str, list[np.ndarray]]
        Interpolated, but unprocessed, data of the independent variable
        varying inside the files.
    y_data_interpolated : defaultdict[str, list[np.ndarray]]
        Interpolated, but unprocessed, data of the dependent variable
        collected from the files.
    x_vary_label : Union[str, Sequence[str]], optional
        Label of the independent variable(s) that varies inside the
        files.
    x_search: Mapping[str, Union[int, float]]
        Dictionary mapping labels of independent variables inside file
        names to their values.
    """
    def __init__(self, x_vary_label: Union[str, Sequence[str]] = None,
                 x_search: Mapping[str, Sequence[Union[int, float]]] = {},
                 y_label: Sequence[str] = None, save_new_labels: bool = False,
                 save_new_values: bool = False, data_type: str = 'admr'):
        self.x_vary_label = x_vary_label
        self.x_search = x_search
        self.y_label = y_label
        self.save_new_labels = save_new_labels
        self.save_new_values = save_new_values
        self.data_type = data_type
        self.x_data = defaultdict(list)
        self.y_data = defaultdict(list)
        self.x_data_raw = defaultdict(list)
        self.y_data_raw = defaultdict(list)
        self.x_data_interpolated = defaultdict(list)
        self.y_data_interpolated = defaultdict(list)

    def __setattr__(self, name, value):
        if name == 'x_search':
            if value != {}:
                values = list(value.values())
                value_len = len(values[0])
                for values in values:
                    if value_len != len(values):
                        raise ValueError("All labels in x_search must have"
                                         " the same number of values.")
        super().__setattr__(name, value)

    def load(self, folder_path: str = '.', prefix: str = '',
             recursive: bool = True,
             x_columns: Union[Sequence[int], Sequence[str]] = None,
             y_columns: Union[Sequence[int], Sequence[str]] = None,
             x_units: Union[Sequence[float], float] = 1.0,
             y_units: Union[Sequence[float], float] = 1.0,
             **kwargs):
        """Load the data from files in the specified folder.

        The function automatically determines which files to load based
        on the given parameters. The file names should be underline
        separated, with each part representing a different
        independent variable. You can put an equal sign or an underline
        between the variable name and its value, but it is not necessary
        if there is no ambiguity. If there is ambiguity, you must put an
        equal sign (an underline is not sufficient). For example, the
        file name can be `admr_lsco_phi=30_field10_rho0=0.1.csv` or
        `admr_lsco_phi_30_field=10_rho0=0.1.csv`.

        Parameters
        ----------
        folder_path : str, optional
            Path to the folder containing the data files.
        prefix : str, optional
            Prefix for the data files.
        recursive : bool, optional
            If True, search for files recursively in the folder and its
            subfolders. If False, search only in the specified folder.
        x_columns : Union[Sequence[int], Sequence[str]], optional
            Override which columns to load for the independent variable.
            If a sequence of integers, it specifies the column indices
            to load. If a sequence of strings, it specifies the column
            names (headers) to load. If not provided, the first column
            is loaded as the independent variable.
        y_columns : Union[Sequence[int], Sequence[str]], optional
            Override which columns to load for the dependent variables.
            If a sequence of integers, it specifies the column indices
            to load. If a sequence of strings, it specifies the column
            names (headers) to load. If not provided, all columns except
            the first one (independent variable) are loaded.
        x_units : Union[Sequence[float], float], optional
            Units for the independent variable(s). If a single float is
            provided, it is applied to all independent variables. If a
            sequence is provided, it must match the number of independent
            variables.
        y_units : Union[Sequence[float], float], optional
            Units for the dependent variable(s). If a single float is
            provided, it is applied to all dependent variables. If a
            sequence is provided, it must match the number of dependent
            variables.
        **kwargs : dict, optional
            Additional keyword arguments to pass to ``numpy.loadtxt``.
        """
        if recursive:
            files = sorted(Path(folder_path).rglob(f"{prefix}*"))
        else:
            files = sorted(Path(folder_path).glob(f"{prefix}*"))
        for file in files:
            if file.is_dir() or not file.name.startswith(prefix):
                continue

            label_map = _extract_labels_and_values(file.name)
            for label, value in label_map.items():
                if self.save_new_labels and label not in self.x_search:
                    self.x_search[label] = []
                if self.save_new_values:
                    self.x_search[label].append(value)

            if self.x_search != {}:
                if not all(label in label_map for label in self.x_search):
                    continue
                if not all(label_map[label] in self.x_search[label]
                           for label in self.x_search):
                    continue
                possible_idx = set()
                for label in self.x_search:
                    if label in label_map:
                        new_idx = [
                            i for i, value in enumerate(self.x_search[label])
                            if value == label_map[label]]
                        if possible_idx == set():
                            possible_idx = set(new_idx)
                        else:
                            possible_idx.intersection_update(new_idx)
                if not possible_idx:
                    continue
                else:
                    idx = min(possible_idx)
            else:
                if self.x_data_raw == {}:
                    idx = 0
                else:
                    first_label = list(self.x_data_raw.keys())[0]
                    idx = len(self.x_data_raw[first_label])

            self._extract_data(file, idx, x_columns, y_columns,
                               x_units, y_units, **kwargs)
        self.process_data()

    def interpolate(self, n_points: int = 50, x_min: float = None,
                    x_max: float = None, x_shift: float = None,
                    x_normalize: float = None, y_shift: float = 0.0,
                    y_normalize: float = 1.0):
        """
        Interpolate the loaded data to the specified number of points.

        For now, only supports linear interpolation for a single
        independent variable varying inside the files.

        Parameters
        ----------
        npoints : int, optional
            Number of points to interpolate to.
        x_min : float, optional
            Lower boundary of the range for the independent variable
            (varying inside the files). If not provided, it is set to
            the minimum value of the independent variable in the loaded
            data.
        x_max : float, optional
            Upper boundary of the range for the independent variable
            (varying inside the files). If not provided, it is set to
            the maximum value of the independent variable in the loaded
            data.
        x_shift : float, optional
            If provided, the data will be shifted by the value at this
            point.
        x_normalize : float, optional
            If provided, the data will be normalized by the value at
            this point. Note that shifts are applied before
            normalization.
        y_shift : float, optional
            The data will be shifted to this value (if ``x_shift``
            is provided).
        y_normalize : float, optional
            The data will be normalized to this value
            (if ``x_normalize`` is provided).
        """
        self.x_data_interpolated = defaultdict(list)
        self.y_data_interpolated = defaultdict(list)
        for i, x in enumerate(self.x_data_raw[self.x_vary_label]):
            x_min = min(x) if x_min is None else x_min
            x_max = max(x) if x_max is None else x_max
            x_new = np.linspace(x_min, x_max, n_points)
            self.x_data_interpolated[self.x_vary_label].append(x_new)
            for y_label, y in self.y_data_raw.items():
                self.y_data_interpolated[y_label].append(
                    np.interp(x_new, x, y[i]))
                if x_shift is not None:
                    y = np.interp(x_shift, x, y[i])
                    self.y_data_interpolated[y_label][-1] -= y
                    self.y_data_interpolated[y_label][-1] += y_shift
                if x_normalize is not None:
                    y = np.interp(x_normalize, x, y[i])
                    self.y_data_interpolated[y_label][-1] /= y
                    self.y_data_interpolated[y_label][-1] *= y_normalize

        self.process_data()

    def process_data(self):
        """Process the loaded data to prepare it for fitting.
        
        Fills the `x_data` and `y_data` attributes with the correct
        values. This is run at the end of the `load` and `interpolate`
        methods. You can use it if you do extra processing on the data
        after loading or interpolating.
        """
        self.x_data = defaultdict(list)
        self.y_data = defaultdict(list)
        if self.y_data_interpolated != {}:
            y_separate = self.y_data_interpolated
        else:
            y_separate = self.y_data_raw
        for label, data in y_separate.items():
            self.y_data[label] = np.concatenate(data)

        if isinstance(self.x_vary_label, str):
            x_vary_label = [self.x_vary_label]
        else:
            x_vary_label = self.x_vary_label
        all_labels = x_vary_label + list(self.x_search.keys())
        x_data_stitched = {label: [] for label in all_labels}
        if self.x_data_interpolated != {}:
            x_separate = self.x_data_interpolated
        else:
            x_separate = self.x_data_raw
        for label in self.x_search:
            for i, value in enumerate(self.x_search[label]):
                x_data_stitched[label].append(
                    np.full(len(x_separate[x_vary_label[0]][i]), value))
        for label in x_vary_label:
            for i, data in enumerate(x_separate[label]):
                while len(x_data_stitched[label]) <= i:
                    x_data_stitched[label].append(np.array([]))
                x_data_stitched[label][i] = data
        x_data_stitched = {label: np.concatenate(data)
                           for label, data in x_data_stitched.items()}
        if self.data_type == 'plain':
            self.x_data = x_data_stitched
        elif self.data_type == 'admr':
            indexing_labels = dict()
            for label in x_data_stitched:
                if label in ['B', 'Bmag', 'Bamp', 'H', 'Hmag', 'Hamp']:
                    indexing_labels['field'] = label
                else:
                    indexing_labels[label.lower()] = label
            field = x_data_stitched[indexing_labels['field']]
            phi = np.deg2rad(x_data_stitched[indexing_labels['phi']])
            theta = np.deg2rad(x_data_stitched[indexing_labels['theta']])
            self.x_data = {'field': field[:, None] * np.column_stack((
                np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                np.cos(theta)))}

    def _extract_data(self, file, idx, x_columns, y_columns,
                      x_units, y_units, **kwargs):
        x_columns, y_columns = self._extract_xy_labels(
            file, x_columns, y_columns, **kwargs)
        data = np.loadtxt(file, **kwargs)

        # "pack" single labels into a list
        if isinstance(self.x_vary_label, str):
            x_vary_label = [self.x_vary_label]
        else:
            x_vary_label = self.x_vary_label
        if isinstance(self.y_label, str):
            y_label = [self.y_label]
        else:
            y_label = self.y_label
        if isinstance(x_units, (int, float)):
            x_units = [x_units] * len(x_columns)
        if isinstance(y_units, (int, float)):
            y_units = [y_units] * len(y_columns)

        for label, col, unit in zip(x_vary_label, x_columns, x_units):
            while len(self.x_data_raw[label]) <= idx:
                self.x_data_raw[label].append(np.array([]))
            self.x_data_raw[label][idx] = unit * data[:, col]
        for label, col, unit in zip(y_label, y_columns, y_units):
            while len(self.y_data_raw[label]) <= idx:
                self.y_data_raw[label].append(np.array([]))
            self.y_data_raw[label][idx] = unit * data[:, col]

    def _extract_xy_labels(self, file, x_columns, y_columns, **kwargs):
        with open(file, 'r') as f:
            for _ in range(kwargs.get('skiprows', 1)):
                line = f.readline()
            while line.startswith(kwargs.get('comments', '#')):
                line = f.readline()
            if self.x_vary_label is None:
                self.x_vary_label = line.split(',')[0].strip()

            headers = [header.strip() for header in line.split(',')]
            if x_columns is None:
                if self.x_vary_label is None:
                    x_columns = [0]
                    self.x_vary_label = headers[0]
                elif isinstance(self.x_vary_label, str):
                    x_columns = [headers.index(self.x_vary_label)]
                else:
                    x_columns = [headers.index(label)
                                 for label in self.x_vary_label]
            elif self.x_vary_label is None:
                self.x_vary_label = [headers[col] for col in x_columns]

            if y_columns is None:
                if self.y_label is None:
                    y_columns = sorted(list(
                        set(range(len(headers))) - set(x_columns)))
                    self.y_label = [headers[col] for col in y_columns]
                elif isinstance(self.y_label, str):
                    y_columns = [headers.index(self.y_label)]
                else:
                    y_columns = [headers.index(label)
                                 for label in self.y_label]
            elif self.y_label is None:
                self.y_label = [headers[col] for col in y_columns]
        return x_columns, y_columns

def _extract_labels_and_values(file_name):
    label_map = {}
    parts = file_name.split('_')
    parts[-1] = '.'.join(parts[-1].split('.')[:-1])
    for i, part in enumerate(parts):
        if '=' in part:
            label, value = part.split('=')
            value = re.search(r'([-+]?\d+\.?\d*)', value).group(0)
            value = float(value)
            if int(value) == value:
                value = int(value)
            label_map[label] = value
        else:
            value = re.search(r'([-+]?\d+\.?\d*[a-zA-Z]*)', part)
            if value:
                value = float(re.search(r'([-+]?\d+\.?\d*)',
                                        value.group(0)).group(0))
                if int(value) == value:
                    value = int(value)
                label = re.search(r'([a-zA-Z_]+)', part)
                if label:
                    label_map[label.group(0)] = value
                elif i > 0:
                    if re.search(r'([a-zA-Z_]+)', parts[i - 1]):
                        label_map[parts[i - 1]] = value
    return label_map
