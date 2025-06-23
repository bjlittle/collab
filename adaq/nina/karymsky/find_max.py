from pathlib import Path

import iris
import numpy as np


cubes = iris.load(Path("data/karymsky_forecast_*.nc"))
vmin, vmax = np.nan, np.nan

for cube in cubes:
    data = cube.data
    dmin = np.min(data)
    dmax = np.max(data)
    print(f"{dmin=}, {dmax=}")
    if np.isnan(vmin):
        vmin = dmin
    elif dmin < vmin:
        vmin = dmin
    if np.isnan(vmax):
        vmax = dmax
    elif dmax > vmax:
        vmax = dmax

print(f"{vmin=}, {vmax=}")