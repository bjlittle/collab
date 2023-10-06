"""Calculate the min/max range of the data payload for the time-series."""
from pathlib import Path

import netCDF4 as nc
import numpy as np


# sort the assets in ascending date order
# set to the appropriate assets directory
path = Path("./assets/seasia")
fnames = list(path.glob("**/*.nc"))
fnames_by_dt = [(int(fname.name.split("_")[0]), fname) for fname in fnames]
fnames = [fname for _, fname in sorted(fnames_by_dt, key=lambda pair: pair[0])]

mins, maxs = [], []

for fname in fnames:
    print(f"{fname} ...")
    ds = nc.Dataset(fname)
    data = ds.variables["toa_outgoing_longwave_flux"][:]
    mins.append(np.min(data))
    maxs.append(np.max(data))

min, max = np.min(mins), np.max(maxs)
print(f"{min=}, {max=}")
