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
        'scattering_models': ['isotropic', 'cos2phi'],
        'scattering_params': {'gamma_0': 12, 'gamma_k': 60, 'power': 12},
        'resolution': 41,
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
