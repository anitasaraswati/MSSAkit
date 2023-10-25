from setuptools import setup, find_packages
import subprocess

REQUIREMENTS = [
    'sys',
    'statsmodels',
    'numpy',    
    'scipy',
    'tqdm',
    'matplotlib',
]


setup(name='mssakit',
      version="0.1",
      description='Blind Source Separation methods',
      author='Anita Thea Saraswati & Olivier de Viron',
      author_email='anitatheasaraswato@gmail.com; olivier.de_viron@univ-lr.fr',
      url='github page',
      packages=find_packages(),
      package_dir={'pymssa':'pymssa'},
      include_package_data=True,
      install_requires=REQUIREMENTS,
      keywords='Multivariate Singular Spectrum Analysis (MSSA) kit')