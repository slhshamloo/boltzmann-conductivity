from .scattering import build_scattering_function
from copy import deepcopy


def easy_params(params):
    """Convenience function to set parameters for the simulation.

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
    new_params = deepcopy(params)
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
