"""Execute script with 'python -i <script>'."""
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

from cf_units import Unit
import geovista
import geovista.theme
from geovista.common import to_cartesian
from geovista.geodesic import BBox
from geovista.geometry import coastlines
from geovista.pantry.meshes import lfric
from geovista.pantry.data import capitalise
from geovista.qt import GeoBackgroundPlotter

import iris
import matplotlib as mpl
import netCDF4 as nc
import numpy as np
import pyvista as pv
from pyvista import Actor
from geovista.geodesic import line


period = "jun22"
# fname = f"xenah/{period}/hartlepool_grid1_C1_202203.nc"
fname = f"xenah/{period}/hartlepool_grid1_C1_202206.nc"
# fname = f"xenah/{period}/hartlepool_grid1_C1_202209.nc"
cube = iris.load_cube(fname)

# bootstrap
t = cube.coord("time")
z = cube.coord("height")
y = cube.coord("latitude")
x = cube.coord("longitude")

unit = Unit(t.units)
fmt = "%Y-%m-%d %H:%M"

ds = nc.Dataset(fname)
payload = ds.variables["xenon_133_air_concentration"]

n_tsteps = t.shape[0]
tstep = 0

y_cb = y.contiguous_bounds()
x_cb = x.contiguous_bounds()
z_cb = z.contiguous_bounds()
print(z_cb)
z_fix = np.arange(*z_cb.shape) * np.mean(np.diff(y_cb)) * 3

xx, yy, zz = np.meshgrid(x_cb, y_cb, z_fix, indexing="ij")
shape = xx.shape
print(f"meshgrid {shape=}")

# vmin, vmax = 0.0, 1.642685  # mar22
# vmin, vmax = 0.0, 0.4848993 # jun22

threshold = 1e-3
vmin = threshold if threshold else 3.2e-6
vmax = 3.2e-2
clim = (vmin, vmax)

cmap = "fire_r"
# cmap = "balance"
color = "white"

xyz = to_cartesian(xx, yy, zlevel=zz, zscale=0.005)
print(f"cartesian {xyz.shape=}")

mesh = pv.StructuredGrid(xyz[:, 0].reshape(shape), xyz[:, 1].reshape(shape), xyz[:, 2].reshape(shape))
data = np.ma.masked_less_equal(payload[tstep][:], 0).filled(np.nan).flatten()
mesh["data"] = data

bbox = BBox([-13.4, 3.4, 3.4, -13.4], [59.2, 59.2, 49.75, 49.75])
coasts = bbox.enclosed(coastlines(zlevel=0))
base = bbox.enclosed(lfric(), preference="point")

pv.global_theme.allow_empty_mesh = True
p = GeoBackgroundPlotter()

p.set_background(color="black")

sargs = {"color": color, "title": f"{capitalise(cube.name())} \n {str(cube.units)}", "font_family": "arial", "label_font_size": 10, "n_labels": 5}

obs = geovista.Transform.from_points(xs=[-0.82637, -1.57331, -1.18083333, -1.56198], ys=[54.55061, 54.76702, 54.635, 53.80476])
p.add_point_labels(obs, ["Boulby", "Durham", "Hartlepool", "Leeds"], render_points_as_spheres=True, point_color="red", point_size=10, font_size=10, text_color="white", justification_horizontal="right", justification_vertical="top", shape=None)

# outline = mesh.extract_feature_edges()
# outline.save(f"vtk/{period}/outline.vtk")
outline = pv.read(f"vtk/{period}/outline.vtk")
p.add_mesh(outline, color="green", line_width=1)

p.add_mesh(coasts, color=color, line_width=0.5)
actor_base = p.add_mesh(base, opacity=0.2)
p.add_mesh(base, style="wireframe", line_width=1, opacity=0.2)
p.add_mesh(bbox.boundary(), color="orange", line_width=3)

dt = t.units.num2date(t.points[tstep])
# tmp = mesh.threshold()
# tmp.save(f"vtk/{period}/mesh_{dt.strftime("%Y%m%d%H%M")}.vtk")
tmp = pv.read(f"vtk/{period}/mesh_{dt.strftime("%Y%m%d%H%M")}.vtk")  #.extract_geometry().smooth_taubin(n_iter=50, pass_band=0.05)
if threshold:
    tmp = tmp.threshold(threshold)
if tmp.n_cells:
    print(f"{tmp.points.max()=}")
actor = p.add_mesh(tmp, scalars="data", name="plume", cmap=cmap, clim=clim, show_edges=False, show_scalar_bar=False)
actor_scalar = p.add_scalar_bar(mapper=actor.mapper, **sargs)
p.add_axes(color=color)


p.show()
