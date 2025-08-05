# MSSAkit
A Python toolkit to analyse time series using Multivariate Singular Spectral Analysis (MSSA)

## Readme and User Guide

The 'MSSAkit' package implements Multivariate Singular Spectrum Analysis (MSSA) as the main function in Python. It also implements the Monte Carlo hypothesis to test the significance of the MSSA decompositions.

The methodology is based on:
Groth, A., Feliks, Y., Kondrashov, D. and Ghil, M., 2017. Interannual variability in the North Atlantic ocean’s temperature field and its association with the wind stress forcing. Journal of Climate, 30(7), pp.2655-2678 (https://doi.org/10.1175/JCLI-D-16-0370.1).

Beside MSSA, this package is also complemented with other functions of source separation, including:
- Singular Spectrum Analysis (SSA)
- Principal Component Analysis (PCA)
- Cross Singular Value Decomposition (cross SVD)

## How to Cite
To cite MSSAkit in publications, please refer to this citation:
Saraswati, A. T., & de Viron, O. (2023). anitasaraswati/MSSAkit: MSSAkit v1.0.0 (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.10377708

And for an example of the extensive use of MSSAkit, please refer also to this article:
Saraswati, A.T., de Viron, O. and Mandea, M., 2023. Earth's core variability from the magnetic and gravity field observations. EGUsphere, 2023, pp.1-34. https://doi.org/10.5194/se-14-1267-2023

## Installation
Install from source:

First, clone this repository from GitHub:
```python
git clone https://github.com/anitasaraswati/MSSAkit.git
cd mssakit
```
And run this line to install:
```python
python setup.py install --single-version-externally-managed --root=/
```
Or replace the last line to install the package using pip:
```python
pip install .
```
If you wish to install the required packages using **conda**, use this line to install MSSAkit:
```python
python setup.py --use-conda install --single-version-externally-managed --root=/
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

The source code us licensed under the [MIT](https://choosealicense.com/licenses/mit/).
