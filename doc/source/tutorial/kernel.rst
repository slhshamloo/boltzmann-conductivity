Beyond RTA: The Scattering Kernel
=================================

When you use a scattering rate to calculate the conductivity, you are assuming that the
scattering only depends on one state at a time. However, in general, if you want to entail the
full intricacies of the scattering process, you need to think of it as a function of two states:
the incoming and outgoing states. This means, instead of a single scattering rate function, you
use a *scattering kernel*.

To actually use a scattering kernel, you have to first decide on a basis that best fits the
symmetries of the problem. For example, for a quasi-2D system, it is a good idea to use
cylindrical harmonics and keep the terms with similar symmetries to the Fermi surface geometry.
To see all available bases, and how to define your own, see the
`API documentation <api/kernel>`_.

To define a scattering kernel, you pass in the nonzero coefficients with a dictionary to
an object of type ``ScatteringKernel``. For example, for a cylindrical harmonic basis,
you can use the ``CylindricalKernel``.

.. code-block:: python

    kernel = elecboltz.CylindricalKernel({
        'constant': 5.5,
        'cos4': 1.0,
        'cos4cos4': 3.0
    })

Here, the 'constant' term is the isotropic scattering rate, the 'cos4' term is a contribution
proportional to :math:`\cos(4\varphi)`, and the 'cos4cos4' term is a contribution proportional
to :math:`\cos(4\varphi)\cos(4\varphi')`, where :math:`\varphi` and :math:`\varphi'` are
azimuthal angles of the outgoing and incoming states, respectively.

To use the kernel for your calculations, you just need to pass it to the conductivity
calculator object instead of a scattering rate, like so:

.. code-block:: python

    conductivity = elecboltz.Conductivity(
        band=band, scattering_kernel=kernel, field=(0, 0, 1.0))

Note that you have to only pass in one of either the scattering rate or the scattering kernel.

You can also use the easy parameter interface for utilizing scattering kernels. For the same
cylindrical harmonics kernel:

.. code-block:: python

    params = elecboltz.easy_params({
        'a': 3.75,
        'b': 3.75,
        'c': 13.2,
        'energy_scale': 160,
        'band_params': {'mu': -0.8243, 't': 1, 'tp':-0.1364,
                        'tpp': 0.0682, 'tz': 0.0651},
        'scattering_kernel_name': 'cylindrical',
        'scattering_kernel_params': {
            'constant': 5.5, 'cos4': 1.0, 'cos4cos4': 3.0},
        'resolution': 41,
        'domain_size': [1.0, 1.0, 2.0],
        'periodic': 2,
        'field': [0.0, 0.0, 1.0]
    })

