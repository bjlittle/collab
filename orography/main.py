from pathlib import Path

import iris
from iris.analysis.cartography import unrotate_pole
import geovista as gv
import geovista.theme
import numpy as np


dname = Path(".")
fname = dname / "umnsaa_pa000.pp"

orog = iris.load_cube(fname, "surface_altitude")

glat = orog.coord("grid_latitude")
glon = orog.coord("grid_longitude")
crs = orog.coord_system()

lon, lat = unrotate_pole(
    glon.points,
    glat.points,
    crs.grid_north_pole_longitude,
    crs.grid_north_pole_latitude,
)

name = "surface_altitude"
mesh = gv.Transform.from_1d(lon, lat, data=orog.data, name="surface_altitude")
mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
mesh.warp_by_scalar(scalars=name, inplace=True, factor=1e-6)

land_mask = iris.load_cube(fname, "land_binary_mask")
land_mask = land_mask.data.astype(np.float32)
land_mask[np.where(land_mask < 1)] = np.nan

mesh.point_data[name] *= np.ravel(land_mask)


sname = " ".join([word.capitalize() for word in name.replace("_", " ").split(" ")])
sargs = dict(title=f"{sname} / {orog.units}", nan_annotation=True, shadow=True)
plotter = gv.GeoPlotter()
plotter.add_text(
    "Falkland Islands Orography", position="upper_left", font_size=10, shadow=True
)
plotter.add_mesh(
    mesh,
    cmap="balance",
    smooth_shading=True,
    preference="point",
    scalars=name,
    scalar_bar_args=sargs,
    show_edges=True,
)
plotter.add_axes()
plotter.show()
