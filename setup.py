#!/usr/bin/env python

from setuptools import setup, find_packages
import sys
import subprocess
import os

REQUIREMENTS = [
    'statsmodels',
    'numpy',    
    'scipy',
    'tqdm',
    'matplotlib',
    'cartopy'
]

if '--use-conda' in sys.argv:
    subprocess.check_call(['conda', 'install', '--file', 'requirements.txt'])
    sys.argv.remove('--use-conda')  # Remove the custom argument

setup(name='mssakit',
      version="0.1",
      description='Blind Source Separation methods',
      author='Anita Thea Saraswati & Olivier de Viron',
      author_email='anitatheasaraswato@gmail.com; olivier.de_viron@univ-lr.fr',
      url='github page',
      packages=find_packages(),
      package_dir={'mssakit':'mssakit'},
      include_package_data=True,
      install_requires=REQUIREMENTS,
      keywords='Multivariate Singular Spectrum Analysis (MSSA) kit')
