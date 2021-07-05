This repository contains the raw data and analysis that are the supporting information to the [`calibr8`](https://github.com/jubiotech/calibr8) paper.

# Contents
The material is organized by the chapter of first appearance in the manuscript.
The `raw_data` directory contains unprocessed result files.

Some of the MCMC trace files (`*.nc`) exceed GitHub's file upload size limit.
We therefore decided to upload the `*.nc` files through [releases](https://github.com/JuBiotech/calibr8-paper/releases).

# Installation
A Python environment for running the notebooks can be created with the following command:

```
conda env create -f environment.yml
```

The new environment is named `murefi_env` and can be activated with `conda activate murefi_env`.

After that a Jupyter notebook server can be launched with `jupyter notebook`.

Note: The only notebook not executable is `0.0 Preprocessing.ipynb`.
This notebook relies on unpublished software. 
However, all processed data is included as XLSX or HDF5 files in the `processed` directory and the data analysis can be re-run accordingly.

# References
* [Preprint on bioRxiv](https://doi.org/10.1101/2021.06.30.450546)
* [`calibr8` source code](https://github.com/JuBiotech/calibr8)
* [`calibr8` documentation](https://calibr8.readthedocs.io)
* [`murefi` source code](https://github.com/JuBiotech/murefi)
* [`murefi` documentation](https://murefi.readthedocs.io)
