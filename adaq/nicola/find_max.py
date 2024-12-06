from pathlib import Path

import iris
import numpy as np


fname = ("data/R2_T2_V2/Fields_grid3_C1.nc")
cube = iris.load_cube(fname)
vmin, vmax = np.nan, np.nan

for i, part in enumerate(cube.slices_over("time")):
    data = part.data
    dmin = np.min(data)
    dmax = np.max(data)
    print(f"\t{i=} {dmax=}")
    if np.isnan(vmin):
        vmin = dmin
    elif dmin < vmin:
        vmin = dmin
    if np.isnan(vmax):
        vmax = dmax
    elif dmax > vmax:
        vmax = dmax

print(f"{vmin=}, {vmax=}")