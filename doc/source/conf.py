import os, sys
from importlib.metadata import PackageNotFoundError, version

sys.path.insert(0, os.path.abspath('../..'))

project = 'elecBoltz'
copyright = '2025, Saleh Shamloo Ahmadi'
author = 'Saleh Shamloo Ahmadi'
try:
    release = version('elecboltz')
except PackageNotFoundError:
    release = '0.0.0'
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
