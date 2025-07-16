Easy Parameter Setting
======================

It is useful to have every parameter in a single dictionary to be able to control
everything easier. Also, having simple key-value pairs for every parameter instead of
constructing each by hand makes setting up pipelines much easier.

To make your parameters out of a single dictionary and somewhat more easily, you can use
``elecboltz.easy_params``. For example:

.. code-block:: python

    import elecboltz
    params = elecboltz.easy_params({
        'a': 3.75,
        'b': 3.75,
        'c': 13.2,
        'energy_scale': 160,
        'band_params': {'mu': -0.8243, 't': 1, 'tp':-0.1364,
                        'tpp': 0.0682, 'tz': 0.0651},
        'scattering_models': ['isotropic', 'cos2phi'],
        'scattering_params': {'gamma_0': 12.595, 'gamma_k': 63.823, 'power': 12.0},
        'resolution': 41,
        'domain_size': [1.0, 1.0, 2.0]
        'periodic': 2
        'field': [0.0, 0.0, 1.0]
    })
    band = elecboltz.BandStructure(**params)
    band.discretize()
    cond = elecboltz.Conductivity(**params)
    cond.calculate()

Here, we have just calculated the coonductivity of Nd-LSCO using a tight-binding model.
By default, ``easy_params`` generates band structures using a tight-binding model
inferred from the parameters given in ``'band_params'``. Since constructing the scattering
function by hand every time can be tedious (especially since there are only a handful of
them that are used frequently), ``'scattering_models'`` and ``'scattering_params'``
are provided to automatically make the functions based on the information.

For more details on ``easy_params``, see the `API documentation <api/params>`_.
