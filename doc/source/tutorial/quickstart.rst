Quickstart Example: Free Electrons
==================================

First, import the module (and anything else you need).

.. code-block:: python

    import elecboltz
    import numpy as np

For a minimal example, let's calculate the conductivity of a simple free electron system.
The dispersion relation for free electrons is given by

.. math::

    E(\mathbf{k}) = \frac{\hbar^2 ||\mathbf{k}||^2}{2m^*}.

For our simple example, ignore the units. Just note that energy units are in milli electronvolts
(meV) and distance units are in angstroms (Å). We can define the dispersion as:

.. code-block:: python

    dispersion = "kx**2 + ky**2 + kz**2"

For this to generate a unit sphere for the Fermi surface in k-space, we need to set the chemical
potential to 1 and the unit cell dimentions to :math:`2\pi` in all three directions.

.. code-block:: python

    mu = 1.0
    cell = (2 * np.pi, 2 * np.pi, 2 * np.pi)

We're now ready to create the band structure object.

.. code-block:: python

    band = elecboltz.BandStructure(
        dispersion=dispersion, chemical_potential=mu, unit_cell=cell,
        periodic=False)

Since we're dealing with a simple sphere, the periodic boundary conditions are not needed.
Next, we can run the discretization to create the mesh used by the solver.

.. code-block:: python

    band.discretize()

Now we can create the conductivity calculator object. Let's just use a scattering rate of 1
and a field of magnitude 1 in the z direction. Note that the scattering rate is in units of
THz (or 1/ps) and the magnetic field is in units of Tesla.

.. code-block:: python

    conductivity = elecboltz.Conductivity(
        band=band, scattering_rate=1.0, field=(0, 0, 1.0))

Finally, we can calculate the conductivity tensor.

.. code-block:: python

    sigma = conductivity.calculate()

That's it! This is the basic workflow for using elecBoltz. Of course, you can do much more;
you can define more complex band structures with various parameters; set the resolution;
and control which axes have periodic boundary conditions;

.. code-block:: python

    dispersion = "- 2*t * (cos(a*kx)+cos(b*ky))" \
                 "- 4*tp * cos(a*kx)*cos(b*ky)" \
                 "- 2*tpp * (cos(2*a*kx)+cos(2*b*ky))" \
                 "- 2*tz * (cos(a*kx)-cos(b*ky))**2" \
                 " * cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)"

    band = elecboltz.BandStructure(
        dispersion=dispersion, chemical_potential=mu, unit_cell=cell,
        band_params={'t': 1, 'tp':-0.13642799, 'tpp':0.06816836, 'tz':0.06512192},
        resolution=[41, 41, 21], periodic=2)
        # periodic in axis 2, which is the z axis (0 is x, 1 is y, 2 is z)

you can have any arbitrary function as the scattering rate;

.. code-block:: python

    def scattering_rate(kx, ky, kz):
        phi = np.atan2(ky, kx)
        return 1.0 + 0.1 * np.abs(np.cos(2*phi))**2

    conductivity = elecboltz.Conductivity(
        band=band, scattering_rate=scattering_rate, field=(0, 0, 1.0))

and even set a frequency for the fields, which gives you optical conductivity.

.. code-block:: python

    conductivity = elecboltz.Conductivity(
        band=band, scattering_rate=1.0, field=(0, 0, 1.0), frequency=1.0)
