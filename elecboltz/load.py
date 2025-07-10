import numpy as np
import re
from typing import Sequence, Union
from pathlib import Path


class Loader:
    """Convenience class for loading data split into multiple files.
    
    Prepares the data for ``fit_model`` by processing multiple files
    labeled with values of independent variables and containing the
    variation of one variable. For example, the files can be labeled
    with the values of ``phi`` and ``field``, and contain the variation
    of ``theta``.

    Parameters
    ----------
    x_vary_label : str, optional
        Label of the independent variable that varies inside the files.
        If not provided, it will be inferred from the file contents
        (column headers).
    x_labels_search : Sequence[str], optional
        Labels of the independent variables to be extracted from
        the file names. If not provided, the value of all labels will
        be extracted.
    x_values_search : Sequence[Sequence[Union[int, float]]], optional
        Which values of the independent variable to load.
        For example, if x_labels_search is ['phi', 'field'] then
        x_values_search should be something like
        [[30, 60], [1.3, 2.7]]. If None, all values corresponding to
        the labels are loaded.
    y_label : Sequence[str], optional
        Labels of the dependent variables, used for fitting routine.
        See ``elecboltz.fit.fit_model`` for details about the allowed
        labels. If None, will be inferred from the file contents
        (column headers).
    extra_labels : Sequence[str], optional
        Labels of extra variables to be extracted from the file names.
    data_type : {'plain', 'admr'}, optional
        Type of data to load. If ``'plain'``, no processing is done on
        the data and the x_data is constructed directly from the
        collected values. If ``'admr'``, the x_data is constructed
        such that it contains the field vector at each index.
    
    Attributes
    ----------
    x_data : Sequence[np.ndarray]
        x_data variable for the fitting procedure.
    y_data : Sequence[np.ndarray]
        y_data variable for the fitting procedure.
    x_label : Sequence[str]
        Labels of the independent variables for fitting.
    x_data_raw : Sequence[np.ndarray]
        Raw data of the independent variable collected from the files.
        Each array corresponds to a different value in
        ``x_values_search``.
    y_data_raw : Sequence[Sequence[np.ndarray]]
        Raw data of the dependent variable collected from the files.
        Each sequence corresponds to a different dependent variable
        in ``y_label``, and each array inside that corresponds
        to a different value in ``x_values_search``.
    x_data_interpolated : Sequence[np.ndarray]
        Interpolated, but unprocessed, data of the independent variable
        varying inside the files.
    y_data_interpolated : Sequence[Sequence[np.ndarray]]
        Interpolated, but unprocessed, data of the dependent variable
        collected from the files.
    extra_values : Sequence[Sequence[float]]
        Values of extra variables extracted from the file names.
        Each sequence corresponds to a different variable in
        `extra_labels`.
    """
    def __init__(self, x_vary_label: str = None,
                 x_labels_search: Sequence[str] = [],
                 x_values_search: Sequence[Union[int, float]] = [],
                 y_label: Sequence[str] = None,
                 extra_labels: Sequence[str] = [], data_type: str = 'admr'):
        self.x_vary_label = x_vary_label
        self.x_labels_search = x_labels_search
        self.x_values_search = x_values_search
        self.y_label = y_label
        self.extra_labels = extra_labels
        self.extra_values = []
        self.data_type = data_type
        self.x_label = []
        self.x_data = []
        self.y_data = []
        self.x_data_raw = []
        self.y_data_raw = []
        self.x_data_interpolated = []
        self.y_data_interpolated = []

    def load(self, folder_path: str = '.', prefix: str = '',
             y_columns: Sequence[int] = None, **kwargs):
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
        y_columns : Sequence[int], optional
            Which columns of the data files to load for the dependent
            variables. If None, all columns are loaded. The number of
            columns should match the length of ``y_label``.
        interpolate : bool, optional
            If True, interpolate the data before post processing.
        **kwargs : dict, optional
            Additional keyword arguments to pass to ``numpy.loadtxt``.
        """
        files = sorted(Path(folder_path).glob(f"{prefix}*"))
        if self.x_values_search == []:
            self._search_all_files(files, prefix, y_columns, **kwargs)
        else:
            self._search_indicated_files(files, prefix, y_columns, **kwargs)
        self.process_data()

    def interpolate(self, n_points: int = 50, x_min: float = None,
                    x_max: float = None):
        """
        Interpolate the loaded data to the specified number of points.

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
        """
        self.x_data_interpolated = [[] for _ in range(len(self.x_data_raw))]
        self.y_data_interpolated = [[] for _ in range(len(self.y_data_raw))]
        for i, x in enumerate(self.x_data_raw):
            x_min = min(x) if x_min is None else x_min
            x_max = max(x) if x_max is None else x_max
            x_new = np.linspace(x_min, x_max, n_points)
            self.x_data_interpolated.append(x_new)
            for y in self.y_data_raw[i]:
                self.y_data_interpolated[i].append(np.interp(x_new, x, y))
        self.process_data()
    
    def process_data(self):
        """Process the loaded data to prepare it for fitting.
        
        Fills the `x_label`, `y_label`, `x_data` and `y_data`
        attributes with the correct values. This is run at the
        end of the `load` and `interpolate` methods. You can use it
        if you do extra processing on the data after loading or
        interpolating.
        """
        self.y_data = []
        for i in range(len(self.y_data_raw)):
            self.y_data.append(np.concatenate(self.y_data_raw[i]))
        all_labels = [self.x_vary_label] + self.x_labels_search
        x_data_stitched = [[] for _ in range(len(self.x_values_search) + 1)]
        x_data_stitched[0] = self.x_data_raw
        for i in range(len(self.x_data_raw)):
            for j in range(len(self.x_values_search)):
                x_data_stitched[j + 1].append(
                    np.full(len(self.x_data_raw[i]),
                            self.x_values_search[j][i]))
        x_data_stitched = [np.concatenate(x) for x in x_data_stitched]
        if self.data_type == 'plain':
            self.x_label = all_labels
            self.x_data = x_data_stitched
        elif self.data_type == 'admr':
            indexing_labels = []
            for label in all_labels:
                if label in ['B', 'Bmag', 'Bamp']:
                    label = 'field'
                indexing_labels.append(label.lower())
            field = x_data_stitched[indexing_labels.index('field')]
            phi = np.deg2rad(x_data_stitched[indexing_labels.index('phi')])
            theta = np.deg2rad(x_data_stitched[indexing_labels.index('theta')])
            self.x_data = [field[:, None] * np.column_stack((
                np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                np.cos(theta)))]
            self.x_label = ['field']

    def _search_all_files(self, files, prefix, y_columns, **kwargs):
        for file in files:
            if not file.name.startswith(prefix):
                continue
            label_map = _extract_labels_and_values(file.name)
            self._sort_labels_and_values(label_map)
            self._extract_data(file, y_columns, **kwargs)

    def _search_indicated_files(self, files, prefix, y_columns, **kwargs):
        for i in range(len(self.x_values_search[0])):
            for file in files:
                if not file.name.startswith(prefix):
                    continue
                for label in self.x_labels_search:
                    if label not in file.name:
                        return True
                label_map = _extract_labels_and_values(file.name)
                if not all(label in label_map
                           for label in self.x_labels_search):
                    continue
                if not all(label in label_map for label in self.extra_labels):
                    continue
                if not all(label_map[label] == self.x_values_search[j][i]
                           for j, label in enumerate(self.x_labels_search)):
                    continue
                self._sort_labels_and_values(label_map)
                self._extract_data(file, y_columns, **kwargs)

    def _sort_labels_and_values(self, label_map):
        add_labels = self.x_labels_search == []
        for label in label_map:
            if label in self.x_labels_search:
                idx = self.x_labels_search.index(label)
                if label_map[label] not in self.x_values_search[idx]:
                    self.x_values_search[idx].append(label_map[label])
            elif label in self.extra_labels:
                if label not in self.extra_labels:
                    self.extra_labels.append(label)
                    self.extra_values.append([])
                idx = self.extra_labels.index(label)
                while idx >= len(self.extra_values):
                    self.extra_values.append([])
                self.extra_values[idx].append(label_map[label])
            elif add_labels:
                self.x_labels_search.append(label)
                self.x_values_search.append([label_map[label]])

    def _extract_data(self, file, y_columns, **kwargs):
        self._extract_none_labels(file, **kwargs)
        if y_columns is None:
            y_columns = list(range(1, len(self.y_label) + 1))
        data = np.loadtxt(file, **kwargs)
        sorted_indices = np.argsort(data[:, 0])
        self.x_data_raw.append(data[sorted_indices, 0])
        if self.y_data_raw == []:
            self.y_data_raw = [[] for _ in self.y_label]
        for i, col in enumerate(y_columns):
            self.y_data_raw[i].append(data[sorted_indices, col])
    
    def _extract_none_labels(self, file, **kwargs):
        with open(file, 'r') as f:
            for _ in range(kwargs.get('skiprows', 1)):
                line = f.readline()
            while line.startswith(kwargs.get('comments', '#')):
                line = f.readline()
            if self.x_vary_label is None:
                self.x_vary_label = line.split(',')[0].strip()
            if self.y_label is None:
                self.y_label = [col.strip() for col in line.split(',')[1:]]


def _extract_labels_and_values(file_name):
    label_map = {}
    parts = file_name.split('_')
    parts[-1] = '.'.join(parts[-1].split('.')[:-1])
    for i, part in enumerate(parts):
        if '=' in part:
            label, value = part.split('=', 1)
            value = float(value)
            if int(value) == value:
                value = int(value)
            label_map[label] = value
        else:
            value = re.search(r'([-+]?\d+\.?\d*)', part)
            if value:
                value = float(value.group(0))
                if int(value) == value:
                    value = int(value)
                label = re.search(r'([a-zA-Z_]+)', part)
                if label:
                    label_map[label.group(0)] = value
                elif i > 0:
                    if re.search(r'([a-zA-Z_]+)', parts[i - 1]):
                        label_map[parts[i - 1]] = value
    return label_map
