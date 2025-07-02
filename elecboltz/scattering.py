import numpy as np
from collections.abc import Collection


def build_scattering_function(
        scattering_params: dict[str, float | Collection[float]],
        scattering_models: Collection[str] = ['isotropic']):
    """
    Build a scattering function from the given parameters.

    Supported scattering models include:

    * 'isotropic': Constant `gamma_0` everywhere
    * | 'cos': `gamma_k * abs(cos(sym * phi))^power` where `phi` is
      | the angle of the projection of the wavevector k in the x-y
      | plane with the x axis. The rest are parameters of the model.
    * | 'sin', 'tan', and 'cot': Same as 'cos' but using different
      | trigonometric functions.
    * | 'cos[n]phi': Where [n] is some integer, e.g. 'cos2phi'. Alias
      | for 'cos' with sym being set to the integer in [n].
    * | 'sin[n]phi', 'tan[n]phi', and 'cot[n]phi': Same as 'cos[n]phi'
      | but using different trigonometric functions.

    Parameters
    ----------
    scattering_params : dict[str, float or Collection[float]]
        Dictionary mapping the names of the parameters to their value
        in each scattering model. If the value is a single number, it
        is assumed to be the parameter for all models. 
    scattering_models : Collection['str'], optional
        The type of scattering model to use.
    
    Returns
    -------
    function
        A callable scattering function.
    """
    def _get_param(key, idx):
        if isinstance(scattering_params[key], (int, float)):
            return scattering_params[key]
        else:
            return scattering_params[key][idx]
    def _get_params(keys, idx):
        return {key: _get_param(key, idx) for key in keys}

    scattering_functions = []
    for i, model in enumerate(scattering_models):
        if model == 'isotropic':
            scattering_functions.append(
                lambda kx, ky, kz: _get_param('gamma_0', i))
        elif model.startswith('cos') or model.startswith('sin'):
            trig_func = {'cos': np.cos, 'sin': np.sin, 'tan': np.tan,
                         'cot': lambda x: 1.0 / np.tan(x)}[model[:3]]
            if len(model) == 3:
                params = _get_params(['gamma_k', 'power'], i)
            else:
                params = _get_params(['gamma_k', 'power'], i)
                params['sym'] = int(model[3:-3])
            scattering_functions.append(
                lambda kx, ky, kz: params['gamma_k'] * np.abs(trig_func(
                    params['sym']*np.atan2(ky, kx)))**params['power'])
    return lambda kx, ky, kz: sum(s(kx, ky, kz) for s in scattering_functions)
