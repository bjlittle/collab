# ⚠️

This directory contains assorted scripts to plot kscale time-series data using `geovista` and `pyvistaqt`.

To create the `conda` environment with the appropriate package dependencies, simply:

```bash
> conda env create --file environment.yml
> conda activate kscale-dev
```

### Assets

The scripts expect the `kscale` time-series data for the `kscale` models to be placed in the following directories:

```bash
assets
├── africa
│   ├── 20200201_20200120T0000Z_africa_olwr_hourly.nc
│   ├── ...
├── global
│   ├── 20200201_20200120T0000Z_global_olwr_hourly.nc
│   ├── ...
├── samerica
│   ├── 20200201_20200120T0000Z_samerica_olwr_hourly.nc
│   ├── ...
├── seasia
│   ├── 20200201_20200120T0000Z_sea_olwr_hourly.nc
│   ├── ...
└── tropical
    ├── 20200201_20200120T0000Z_channel_olwr_hourly.nc
    ├── ...

```

The `clim.py` script is a convenience to calculate the colormap range `(minimum, maximum)` over the time-series
data of a specific model stored in the `assets` directory.

Note that, the `assets` are not under version control, due to their data volume.

### Scripts

The scripts are proof-of-concept and are taylored for specific model data. At this point, no attempt has been made to
make the scripts generic.

The scripts use the QT bindings to `pyvista` to render real-time animations of the model data over the time-series using
`geovista`.

Therefore, to execute a script please ensure to do so using the Python interactive mode `-i`, e.g.,

```bash
> python -i main-ts-global.py
```

Note that, the scripts assume that there are 24 hourly time steps in each netCDF file i.e., each netCDF file is for 1 day.

The scripts make use of an `opacity` transfer function, therefore `opacity` support is required by the underlying MESA.