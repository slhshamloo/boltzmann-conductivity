# FEM for Boltzmann Transport Conductivity
Conductivity calculated using Boltzmann Transport theory, using a Finite Element Method (FEM).
In addition to arbitrary Fermi surfaces, arbitrary scattering kernels are also supported.

Developed and maintained by the [Grissonnanche group](https://gaelgrissonnanche.com).

## Installation
```
pip install elecboltz
```

## Usage
Here is a minimal example for a simple free electron system.
```python
import elecboltz

band = elecboltz.BandStructure(
    dispersion="kx**2 + ky**2", chemical_potential=1.0,
    unit_cell=(2*np.pi, 2*np.pi, 2*np.pi), periodic=2)
band.discretize()

cond = elecboltz.Conductivity(
    band=band, scattering=1.0, field=(1.0, 2.0, 3.0))
sigma = cond.solve()
print(sigma)
```

## Documentation
The documentation is available at [elecboltz.readthedocs.io](https://elecboltz.readthedocs.io).
