# MSSAkit
A Python toolkit to analyse time series using Multivariate Singular Spectral Analysis (MSSA)

# Readme and User Guide

The 'MSSAkit' package implements Multivariate Singular Spectrum Analysis (MSSA) as the main function in python. It implements also the Monte Carlo hypothesis to test the significance of the MSSA decompositions.

The methodology is based on:
Groth, A., Feliks, Y., Kondrashov, D. and Ghil, M., 2017. Interannual variability in the North Atlantic ocean’s temperature field and its association with the wind stress forcing. Journal of Climate, 30(7), pp.2655-2678 (https://doi.org/10.1175/JCLI-D-16-0370.1).

Beside MSSA, this package is also complemented with other functions of source separation, including:
- Singular Spectrum Analysis (SSA)
- Principal Component Analysis (PCA)
- Cross Singular Value Decomposition (cross SVD)

# HOW TO CITE
To cite MSSAkit in publications, please refer to this article:
Saraswati, A.T., de Viron, O. and Mandea, M., 2023. Earth's core variability from the magnetic and gravity field observations. EGUsphere, 2023, pp.1-34.

## Installation

Install from source:: 
'''bash
    $ git clone https://github.com/zzheng93/pyEOF.git
    $ cd mssakit
    $ python setup.py install
'''

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

The source code us licensed under the [MIT](https://choosealicense.com/licenses/mit/).
