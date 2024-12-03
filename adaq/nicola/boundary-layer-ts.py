"""Execute script with 'python -i <script>'."""

from datetime import datetime
from functools import partial
from pathlib import Path

from cf_units import Unit
import geovista
import geovista.theme
from geovista.common import to_cartesian
from geovista.core import add_texture_coords
from geovista.geodesic import BBox
from geovista.geometry import coastlines
from geovista.pantry.meshes import regular_grid
from geovista.pantry.data import capitalise
from geovista.qt import GeoBackgroundPlotter
from geovista.raster import wrap_texture

import iris
import netCDF4 as nc
import pyvista as pv


# def callback(value) -> None:
def callback() -> None:
    global tstep
    global n_tsteps
    global t
    global cmap
    global clim
    global mesh
    global actor_text
    global actor_scalar
    global factor
    global opacity

    tstep = (tstep + 1) % n_tsteps

    # if tstep == 0:
        # exit()

    dt = t.units.num2date(t.points[tstep])

    # mesh["data"] = data[tstep][:].flatten()
    # mesh.save(f"vtk/mesh_boundary_layer_{dt.strftime("%Y%m%d%H%M")}.vtk")
    mesh = pv.read(f"vtk/mesh_boundary_layer_{dt.strftime('%Y%m%d%H%M')}.vtk")
    mesh = mesh.cell_data_to_point_data()
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True, flip_normals=True)
    mesh.warp_by_scalar(scalars="data", inplace=True, factor=factor)
    # mesh.smooth_taubin(inplace=True)
    mesh.active_scalars_name = "data"

    p.add_mesh(mesh, name="boundary", opacity=opacity, cmap=cmap, clim=clim, render=False, reset_camera=False, show_scalar_bar=False)
    p.add_actor(actor_scalar)

    text = f"{dt.strftime(fmt)}"
    actor_text.SetText(3, text)


fname = "data/Boundary_Layer_C1.nc"
cube = iris.load_cube(fname)

# bootstrap
t = cube.coord("time")
y = cube.coord("latitude")
x = cube.coord("longitude")

fmt = "%Y-%m-%d %H:%M"

ds = nc.Dataset(fname)
data = ds.variables["var__boundary_layer_depth"]

n_tsteps = t.shape[0]
tstep = 0

y_cb = y.contiguous_bounds()
x_cb = x.contiguous_bounds()

threshold = None
vmin = threshold if threshold else 50.0
vmax = 640.0  # 639.9715
clim = (vmin, vmax)

# cmap = "fire_r"
# cmap = "balance"
cmap = "deep_r"
color = "white"

factor = 4e-6

# mesh = geovista.Transform.from_1d(x_cb, y_cb)
# mesh["data"] = data[tstep][:].flatten()

dt = t.units.num2date(t.points[tstep])
# mesh.save(f"vtk/mesh_boundary_layer_{dt.strftime("%Y%m%d%H%M")}.vtk")
mesh = pv.read(f"vtk/mesh_boundary_layer_{dt.strftime('%Y%m%d%H%M')}.vtk")
mesh = mesh.cell_data_to_point_data()
mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True, flip_normals=True)
mesh.warp_by_scalar(scalars="data", inplace=True, factor=factor)
# mesh.smooth_taubin(inplace=True)

bbox = BBox([-1.0, 0.9, 0.9, -1.0], [52.9, 52.9, 51.5, 51.5])    # [52.7, 52.7, 51.7, 51.7])
grid = regular_grid("r1000")
base = bbox.enclosed(grid, preference="point")
base = add_texture_coords(base)
texture = wrap_texture(geovista.natural_earth_hypsometric())

p = GeoBackgroundPlotter()

# p.enable_anti_aliasing(aa_type="ssaa")
p.enable_lightkit()
p.set_background(color="black")

opacity = None
actor_boundary = p.add_mesh(mesh, name="boundary", opacity=opacity, cmap=cmap, clim=clim, show_edges=False, show_scalar_bar=False)
p.view_poi()

sargs = {"color": color, "title": f"{capitalise(cube.name())} / {str(cube.units)}", "font_family": "arial", "label_font_size": 10, "n_labels": 5}
actor_scalar = p.add_scalar_bar(mapper=actor_boundary.mapper, **sargs)

p.add_mesh(base, texture=texture, opacity=0.5, show_edges=True)

obs = geovista.Transform.from_points(xs=[0.1313, -0.2408], ys=[52.1951, 52.5703])
p.add_point_labels(obs, ["Cambridge", "Peterborough"], text_color=color, render_points_as_spheres=True, point_color="red", point_size=10, font_size=10, justification_horizontal="right", justification_vertical="top", shape=None)

text = f"{dt.strftime(fmt)}"
actor_text = p.add_text(
    text, position="upper_right", font_size=10, color=color, shadow=False
)
p.add_text(
    f"Boundary Layer",
    position="upper_left",
    font_size=10,
    color=color,
    shadow=False
)
p.add_axes(color=color)

del cube

input()

p.add_callback(callback, interval=200)