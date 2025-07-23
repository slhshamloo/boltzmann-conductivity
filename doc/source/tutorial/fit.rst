Fitting
=======

The workflow of transport research is centered around fitting a model to experimental data.
This package provides convenient tools for this exact task to make the process as smooth as
possible.

Loading data
------------

First, you can load your data from various files separated by the different variables of the
experiment. For example, for ADMR, suppose you have a file for every combination of field and
azimuthal angle (:math:`\phi`) and the polar angle (:math:`\theta`) varies in each file.
Then, your data folder structure should look something like this:

.. code-block:: text

    data/
        ADMR_10072025_phi=0_B=2.csv
        ADMR_10072025_phi=0_B=5.csv
        ADMR_11072025_phi=15_B=2.csv
        ADMR_11072025_phi=15_B=5.csv
        ADMR_12072025_phi=30_B=2.csv
        ADMR_12072025_phi=30_B=5.csv
        ADMR_12072025_phi=45_B=2.csv
        ADMR_12072025_phi=45_B=5.csv

It might also be separed into subfolders. The important thing is that the file names contain
all the information about the experiment. With such data, you can use ``elecboltz.Loader``
to automatically load the data into arrays usable by the fitting functions of the package.

.. code-block:: python

    import elecboltz

    loader = elecboltz.Loader(
        x_vary_label='theta', y_label='rho_zz',
        x_search={'phi': [0, 0, 15, 15, 30, 30, 45, 45],
                  'B': [2, 5, 2, 5, 2, 5, 2, 5]})
    loader.load("data/ADMR_NdLSCO", "ADMR_",
                x_columns=[0], y_columns=[1], y_units=1e-5)

You can also let the loader find the values by itself using the ``save_new_labels`` and
``save_new_values`` parameters.

Now, fitting all the data at once is goint to be very computationally expensive. Also,
we might want to normalize the data first before fitting. We can do both using the
``interpolate`` method of the loader.

.. code-block:: python

    loader.interpolate(30, x_normalize=0)

This will interpolate each data set to 30 points, and normalize the data by the value at x=0.
For our ADMR example, this would be at :math:`\theta=0^\circ`.

To learn more about the loader, check the :ref:`Loader's API documentation <load>`:.

Fitting Routine
---------------

After loading the data, you can set the ranges and the initial parameters and feed it
into the fitter.

.. code-block:: python

    init_params = {
        'a': 3.75,
        'b': 3.75,
        'c': 13.2,
        'energy_scale': 160,
        'band_params': {'mu':-0.82439881, 't': 1, 'tp':-0.13642799,
                        'tpp':0.06816836, 'tz':0.06512192},
        'domain_size': [1.0, 1.0, 2.0],
        'periodic': 2,
        'resolution': 41,
        'scattering_models': ['isotropic', 'cos2phi'],
        'scattering_params': {'gamma_0': 12, 'gamma_k': 60, 'power': 12},
    }

    bounds = {
        'band_params': {
            'tz': (0.01, 0.1)
        },
        'scattering_params': {
            'gamma_0': (5, 20),
            'gamma_k': (10, 200),
            'power': (1, 20)
        },
    }

    elecboltz.fit_model(
        loader.x_data, loader.y_data, init_params=init_params,
        bounds=bounds, save_path="fit", save_label="ADMR", workers=4)

That's it! The output will be saved to the ``fit/ADMR.json`` file, and the logs will be
saved to ``fit/ADMR.log`` and also printed to the console.

To learn more about the fitting routine, check the :ref:`Fitter's API documentation <fit>`:.

Multi-parameter Fitting
-----------------------

Sometimes, you might want to vary some fitting parameters over a specific parameter, but keep
other fitting parameters the same over that parameter. For example, when finding the temperature
dependence of scattering parameters, you should fit the same band parameters for all temperatures,
while fitting different scattering parameters for each temperature.

To do this type of multi-parameter fitting, you can use the ``split_by`` parameter of the loader
to choose the parameter over which different parameters might be fitted differently (e.g. the
temperature in the example), and then use the ``multi_params`` parameter of the ``fit_model``
function to specify the parameters that should be fitted differently over that parameter (e.g.
the scattering parameters in the example). Then, you can also set different initial values and
bounds for the multi-parameters for each value of the split parameter (e.g. different for each
temperature). You can also still set a single value for all of them.

For this example, assume the files are structured like this:

.. code-block:: text

    data/
        ADMR_10072025_phi=0_T=2.csv
        ADMR_10072025_phi=0_T=5.csv
        ADMR_11072025_phi=15_T=2.csv
        ADMR_11072025_phi=15_T=5.csv
        ADMR_12072025_phi=30_T=2.csv
        ADMR_12072025_phi=30_T=5.csv
        ADMR_12072025_phi=45_T=2.csv
        ADMR_12072025_phi=45_T=5.csv

Then, you can do a multi-parameter fit with parameters varying over the temperature like this:

.. code-block:: python

    import elecboltz

    loader = elecboltz.Loader(
        x_vary_label='theta', y_label='rho_zz',
        x_search={'phi': [0, 0, 15, 15, 30, 30, 45, 45],
                  'B': [2, 5, 2, 5, 2, 5, 2, 5]})
    loader.load("data/ADMR_NdLSCO", "ADMR_",
                x_columns=[0], y_columns=[1], y_units=1e-5)

    init_params = {
        'a': 3.75,
        'b': 3.75,
        'c': 13.2,
        'energy_scale': 160,
        'band_params': {'mu':-0.82439881, 't': 1, 'tp':-0.13642799,
                        'tpp':0.06816836, 'tz':0.06512192},
        'domain_size': [1.0, 1.0, 2.0],
        'periodic': 2,
        'resolution': 41,
        'scattering_models': ['isotropic', 'cos2phi'],
        'scattering_params': {
            'gamma_0': 12,
            'gamma_k': [60, 60, 60, 60],
            'power': [12, 12, 12, 12]
        },
    }

    bounds = {
        'band_params': {
            'tz': (0.01, 0.1)
        },
        'scattering_params': {
            'gamma_0': [(5, 20), (5, 20), (5, 20), (5, 20)],
            'gamma_k': (10, 200),
            'power': [(1, 20), (1, 20), (1, 20), (1, 20)],
        },
    }
