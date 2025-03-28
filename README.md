# JWST Spectroscopic Analysis Toolkit

This repository contains a collection of Python functions for analyzing astronomical data. The functions provided here focus on spectral analysis and calculations related to galaxy properties. Below is an overview of the functionalities provided by this toolkit.

## Features

### Spectrum Analysis

1. **Read Spectrum**: Read 1D spectra from FITS files using `specutils`.
2. **Load 2D Spectrum**: Load 2D spectra from FITS files.
3. **Deredshift Spectrum**: De-redshift the spectrum to its rest frame.
4. **Create Rest-frame Spectrum**: Generate a rest-frame spectrum.
5. **Measure UV Slope (Beta)**: Calculate the UV slope beta of the spectrum.
6. **Measure UV Slope with Bootstrapping**: Estimate UV slope beta with bootstrapping for error estimation.
7. **Measure Equivalent Width (EW)**: Compute the equivalent width of a spectral line.

### Photometric Analysis

1. **Calculate 1500 Magnitude**: Determine the 1500 magnitude of a spectrum given its redshift.

### Nebular Line Analysis

1. **Calculate Intrinsic Ratios**: Compute intrinsic ratios between H lines.
2. **Calculate Reddening Curves**: Determine reddening curves at wavelengths of interest, assuming the Small Magellanic Cloud (SMC) extinction law.
3. **Calculate E(B-V)**: Estimate the reddening (E(B-V)) based on observed H line fluxes.
4. **Calculate SFRs using a couple of provided prescriptions**: Estimating the SFR using the Kennicutt relation for single-stars and a new relation for lower metallicities and binary stars.
5. **Calculate Lyman-alpha Escape Fraction**: Determine the Lyman-alpha escape fraction.
6. **Calculate Ionizing Photon Production Efficiency (xi_ion)**: Compute the ionizing photon production efficiency.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Astropy
- Specutils
- MPDAF
- PyNeb
- Dust_Extinction

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aayush3009/measure_spec.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To use any given function/API, first import spec_analysis as:
```bash
from measure_spec import spec_analysis as ms
```
and then to use the function, do:
```bash
ms.measure_beta_bootstrap(*args)
```

Each Python file contains a set of functions related to a specific aspect of JADES spectroscopic analysis. You can import these functions into your Python environment or scripts as needed. Make sure to provide proper inputs to each function as per the documentation provided.

## Contributing

Contributions to this toolkit are welcome! If you encounter any bugs or have suggestions for new features, please open an issue or submit a pull request.
