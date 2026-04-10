import os, sys
from setuptools_scm import get_version

sys.path.insert(0, os.path.abspath('../..'))

project = 'elecBoltz'
copyright = '2025, Saleh Shamloo Ahmadi'
author = 'Saleh Shamloo Ahmadi'

release = get_version(root='../../')
version = release.split("+")[0]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
