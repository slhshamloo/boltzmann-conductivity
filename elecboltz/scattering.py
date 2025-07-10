import numpy as np
from typing import Union, Callable
from collections.abc import Sequence


class ScatteringFunction:
    """A class to represent a scattering function.

    Abstract base class for scattering functions that can be
    evaluated at a given wavevector (kx, ky, kz). Subclasses should
    implement the `__call__` method.
    """
    def __init__(self, params):
        self.params = params
    
    def __call__(self, kx, ky, kz, **kwargs):
        """Evaluate the scattering function at the given wavevector.

        Parameters
        ----------
        kx, ky, kz : float
            The components of the wavevector in Cartesian coordinates.

        Returns
        -------
        float
            The value of the scattering function at
            the given wavevector.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class IsotropicScattering(ScatteringFunction):
    """A class for isotropic scattering.

    This class represents a scattering function that is constant
    everywhere, defined by the parameter `gamma_0`.
    """
    def __call__(self, kx, ky, kz, **kwargs):
        return self.params['gamma_0']


class AzimuthalScattering(ScatteringFunction):
    """A class for azimuthal scattering.

    This class represents a scattering function that depends on the
    azimuthal angle of the wavevector in the x-y plane, defined by
    parameters `gamma_k`, `power`, `trig`, and `sym`. The scattering
    function is defined as `gamma_k * abs(trig(sym * phi))^power`,
    where `phi` is the angle of the projection of the wavevector k in
    the x-y plane with the x axis, and `trig` is a trigonometric
    function (cos, sin, tan, or cot) depending on the `trig` parameter.
    """
    def __init__(self, params):
        super().__init__(params)
        trig_funcs = {'cos': np.cos, 'sin': np.sin, 'tan': np.tan, 'cot': _cot}
        self.trig_func = trig_funcs[params['trig']]
        if 'sym' not in params:
            self.params['sym'] = 1
    def __call__(self, kx, ky, kz, **kwargs):
        phi = np.arctan2(ky, kx)
        return (self.params['gamma_k'] * np.abs(self.trig_func(
            self.params['sym']*phi)) ** self.params['power'])


class ScatteringSum(ScatteringFunction):
    """A class for summing multiple scattering functions.

    This class represents a scattering function that is the sum of
    multiple scattering functions, each defined by its own parameters.
    It allows for combining different scattering models into a single
    function.
    """
    def __init__(self, scattering_functions):
        super().__init__(params={})
        self.scattering_functions = scattering_functions
    
    def __call__(self, kx, ky, kz, **kwargs):
        return sum(s(kx, ky, kz, **kwargs) for s in self.scattering_functions)


def build_scattering_function(
        scattering_params: dict[str, Union[float, Sequence[float]]],
        scattering_models: Sequence[str] = ['isotropic']):
    """Build a scattering function from the given parameters.

    Supported scattering models include:

    * ``'isotropic'``: Constant ``gamma_0`` everywhere
    * | ``'cos'``: ``gamma_k * abs(cos(sym * phi))^power`` where `phi`
      | is the angle of the projection of the wavevector k in the x-y
      | plane with the x axis. The rest are parameters of the model.
    * | ``'sin'``, ``'tan'``, and ``'cot'``: Same as ``'cos'`` but
      | using different trigonometric functions.
    * | ``'cos[n]phi'``: Where [n] is some integer, e.g. ``'cos2phi'``.
      | Alias for ``'cos'`` with sym being set to the integer in [n].
    * | ``'sin[n]phi'``, ``'tan[n]phi'``, and ``'cot[n]phi'``: Same as
      | ``'cos[n]phi'`` but using different trigonometric functions.

    Parameters
    ----------
    scattering_params : dict[str, float or Sequence[float]]
        Dictionary mapping the names of the parameters to their value
        in each scattering model. If the value is a single number, it
        is assumed to be the parameter for all models. 
    scattering_models : Sequence['str'], optional
        The type of scattering model to use.
    
    Returns
    -------
    Callable
        A class that implements the scattering function and is callable
        on (kx, ky, kz).
    """

    scattering_functions = []
    for i, model in enumerate(scattering_models):
        if model == 'isotropic':
            scattering_functions.append(IsotropicScattering(
                {'gamma_0': _get_param(scattering_params, 'gamma_0', i)}))
        elif model.startswith('cos') or model.startswith('sin'):
            params = _get_params(scattering_params, ['gamma_k', 'power'], i)
            params['trig'] = model[:3]
            if len(model) > 3:
                params['sym'] = int(model[3:-3])
            scattering_functions.append(AzimuthalScattering(params))
    return ScatteringSum(scattering_functions)


def _get_param(scattering_params, key, idx):
        if isinstance(scattering_params[key], (int, float)):
            return scattering_params[key]
        else:
            return scattering_params[key][idx]


def _get_params(scattering_params, keys, idx):
    return {key: _get_param(scattering_params, key, idx) for key in keys}


def _cot(x):
    """Compute the cotangent of x. Defined to avoid lambda function"""
    return 1.0 / np.tan(x) if x != 0 else np.inf
