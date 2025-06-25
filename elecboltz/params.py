import numpy as np


def easy_params(params):
    """
    Convenience function to set parameters for the simulation.

    List of convenience features:

    * | Set unit cell dimensions with named parameters `a`, `b`, and
      | `c`. If `b` or `c` are not given, they are assumed to be equal
      | to `a`.
    * Define an `energy_scale` which scales all energy parameters.
    * | Set the chemical potential with `mu` in `band_params`. If `mu`
      | is set in `band_params`, it is assumed the energy dispersion is
      | shifted by the chemical potential, so the associated variable
      | is set to 0.0 in the returned parameters.
    * | Set the dispersion relation with a default tight-binding model.
      | See `get_tight_binding_dispersion` for the list of parameters
      | and the resulting expression.
    * | Build the scattering function using predefined
      | `scattering_models` and the `scattering_params` associated with
      | them. See `build_scattering_function` for supported scattering
      | models and their parameters. `scattering_models` is assumed to
      | be only one `isotropic` model if not specified.

    Parameters
    ----------
    params : dict
        Simplified (easy-to-use) parameters for the simulation.
    
    Returns
    -------
    dict
        Parameters compatible with the classes in the package.
    """
    new_params = params.copy()
    # unit cell dimensions indicated by axis names
    if 'a' in params:
        unit_cell = [params['a'], params['a'], params['a']]
        if 'b' in params:
            unit_cell[1] = params['b']
        if 'c' in params:
            unit_cell[2] = params['c']
        new_params['unit_cell'] = unit_cell
    # some like to shift the energy dispersion itself by the chemical
    # potential and treat similarly to the other energy parameters
    if 'band_params' in params:
        new_params['band_params'] = params['band_params'].copy()
        if 'mu' in params['band_params']:
            new_params['chemical_potential'] = 0.0
    # scale all energy parameters by a given factor
    if 'energy_scale' in params:
        for key in new_params['band_params']:
            new_params['band_params'][key] *= params['energy_scale']
    # get the default tight-binding dispersion relation
    if 'dispersion' not in new_params:
        new_params['dispersion'] = get_tight_binding_dispersion(
            new_params['band_params'])
    # automatically build the scattering function from named parameters
    if 'scattering_params' in params:
        if 'scattering_models' not in params:
            new_params['scattering_models'] = ['isotropic']
        new_params['scattering_rate'] = build_scattering_function(
            new_params['scattering_params'], new_params['scattering_models'])
    return new_params


def get_tight_binding_dispersion(band_params) -> str:
    """
    Get the tight-binding dispersion relation containing terms relating
    to the parameters in `band_params`.

    The full tight-binding dispersion relation is given by::

        -mu - 2*t * (cos(a*kx)+cos(b*ky))
        - 4*tp * cos(a*kx)*cos(b*ky)
        - 2*tpp * (cos(2*a*kx)+cos(2*b*ky))
        - 2*tz * (cos(a*kx)-cos(b*ky))**2
            * cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)

    The list of parameters is as follows:

    * mu: Chemical potential.
    * t: Nearest-neighbor hopping parameter in the x-y plane.
    * tp: Next-nearest-neighbor hopping parameter in the x-y plane.
    * | tpp: Next-next-nearest-neighbor hopping parameter
      | in the x-y plane.

    * | tz: Nearest-neighbor hopping parameter between
      | the different layers in the z direction.

    Parameters
    ----------
    band_params : dict or set
        Dictionary or set of parameters for the tight-binding model.
    
    Returns
    -------
    str
        The dispersion relation expression string
    """
    dispersion = ""
    if 'mu' in band_params:
        dispersion += "-mu"
    if 't' in band_params:
        dispersion += "-2*t*(cos(a*kx)+cos(b*ky))"
    if 'tp' in band_params:
        dispersion += "-4*tp*cos(a*kx)*cos(b*ky)"
    if 'tpp' in band_params:
        dispersion += "-2*tpp*(cos(2*a*kx)+cos(2*b*ky))"
    if 'tz' in band_params:
        dispersion += "-2*tz*(cos(a*kx)-cos(b*ky))**2"
        dispersion += "*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)"
    return dispersion


def build_scattering_function(
        scattering_params, scattering_models=['isotropic']):
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
