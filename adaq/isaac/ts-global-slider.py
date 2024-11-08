"""Execute script with 'python -i <script>'."""
from datetime import datetime, timedelta
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
import matplotlib as mpl
import netCDF4 as nc
import numpy as np
import pyvista as pv
from pyvista import Actor
from geovista.geodesic import line


show_slice = False


# def callback(value) -> None:
def callback() -> None:
    global tstep
    global n_tsteps
    global payload
    global mesh
    global fmt
    global t
    global unit
    global actor_text
    global clim
    global p
    global cmap
    global sargs
    global vmin
    global period
    global actor_scalar
    global threshold
    global show_slice

    # value = int(value)

    # tstep = value % n_tsteps
    tstep = (tstep + 1) % n_tsteps

    if tstep == 0:
        tstep = 1

    print(f"{tstep=}")

    dt = t.units.num2date(t.points[tstep])
    # mesh["data"] = np.ma.masked_less_equal(payload[tstep][:], 0).filled(np.nan).flatten()
    # mesh.active_scalars_name = "data"
    # tmp = mesh.threshold()
    # tmp.save(f"vtk/mesh_grid1_{dt.strftime("%Y%m%d%H%M")}.vtk")
    tmp = pv.read(f"vtk/mesh_grid1_{dt.strftime('%Y%m%d%H%M')}.vtk")

    # tmp = tmp.extract_geometry().smooth_taubin(n_iter=50, pass_band=0.05, normalize_coordinates=True, feature_angle=30, non_manifold_smoothing=True)

    if threshold:
        tmp = tmp.threshold(threshold)

    # if tmp.n_cells:
    #     print(f"{tmp['data'].min()}, {tmp["data"].max()}, {tmp['data'].mean()}, {np.median(tmp['data'])}")

    p.add_mesh(tmp, name="plume", cmap=cmap, clim=clim, render=False, reset_camera=False, show_scalar_bar=False, above_color="green")
    p.add_actor(actor_scalar)

    text = f"{dt.strftime(fmt)}"
    actor_text.SetText(3, text)

    # if tstep == 0:
    #     exit()


def callback_base(actor: Actor, flag: bool) -> None:
    actor.SetVisibility(flag)


fname_plume = f"fukushima_grid1_201103.nc"
plume = iris.load_cube(fname_plume)

# bootstrap
t = plume.coord("time")
z = plume.coord("height")
y = plume.coord("latitude")
x = plume.coord("longitude")

unit = Unit(t.units)
fmt = "%Y-%m-%d %H:%M"

# ds = nc.Dataset(fname_plume)
# payload = ds.variables["CS137_AIR_CONCENTRATION"]

# n_tsteps = t.shape[0]
n_tsteps = 290
tstep = 1

# y_cb = y.contiguous_bounds()
# x_cb = x.contiguous_bounds()
# z_cb = z.contiguous_bounds()
# print(z_cb)
# z_fix = np.arange(*z_cb.shape) * np.mean(np.diff(y_cb)) * 4
#
# xx, yy, zz = np.meshgrid(x_cb, y_cb, z_fix, indexing="ij")
# shape = xx.shape
# print(f"meshgrid {shape=}")

threshold = 1e1
vmin = threshold if threshold else 0
vmax = 500.0
clim = (vmin, vmax)

cmap = "fire_r"
# cmap = "balance"
color = "white"

# xyz = to_cartesian(xx, yy, zlevel=zz, zscale=0.005)
# print(f"cartesian {xyz.shape=}")
#
# mesh = pv.StructuredGrid(xyz[:, 0].reshape(shape), xyz[:, 1].reshape(shape), xyz[:, 2].reshape(shape))
# data = np.ma.masked_less_equal(payload[tstep][:], 0).filled(np.nan).flatten()
# mesh["data"] = data

pv.global_theme.allow_empty_mesh = True
p = GeoBackgroundPlotter()

# p.enable_anti_aliasing(aa_type="ssaa")
p.enable_lightkit()

bbox = BBox(lons=[132.0, 149.4, 149.4, 132.0], lats=[46.5, 46.5, 30.0, 30.0])
coasts = bbox.enclosed(coastlines(resolution="10m", zlevel=0))
base = add_texture_coords(bbox.enclosed(regular_grid("r400"), preference="center"))
edge = base.extract_feature_edges()
texture = wrap_texture(geovista.natural_earth_hypsometric())

p.set_background(color="black")

sargs = {"color": color, "title": f"{capitalise(plume.name())} \n {str(plume.units)}", "font_family": "arial", "label_font_size": 10, "n_labels": 5}

actor_base = p.add_mesh(base, texture=texture, opacity=0.3, show_edges=True)
p.view_poi()
p.add_mesh(coasts, color="lightgray")
p.add_mesh(edge, color="lightgray")
# p.add_mesh(bbox.boundary(), color="orange", line_width=2)

obs = geovista.Transform.from_points(xs=[141.03305556], ys=[37.42305556])
p.add_point_labels(obs, ["Fukushima\nDaiichi   "], render_points_as_spheres=True, point_color="red", point_size=10, font_size=10, text_color="white", justification_horizontal="right", justification_vertical="top", shape=None)

# outline = mesh.extract_feature_edges()
# outline.save(f"vtk/outline.vtk")
outline = pv.read(f"vtk/outline.vtk")
p.add_mesh(outline, color="orange", line_width=1)

dt = t.units.num2date(t.points[tstep])
# tmp = mesh.threshold()
# tmp.save(f"vtk/mesh_grid1_{dt.strftime("%Y%m%d%H%M")}.vtk")
tmp = pv.read(f"vtk/mesh_grid1_{dt.strftime('%Y%m%d%H%M')}.vtk")

# tmp = tmp.extract_geometry().smooth_taubin(n_iter=50, pass_band=0.05, normalize_coordinates=True, feature_angle=30, non_manifold_smoothing=True)

if threshold:
    tmp = tmp.threshold(threshold)

# if tmp.n_cells:
#     print(f"{tmp['data'].min()}, {tmp['data'].max()}, {tmp['data'].mean()}, {np.median(tmp['data'])}")

actor = p.add_mesh(tmp, scalars="data", name="plume", cmap=cmap, clim=clim, show_scalar_bar=False, above_color="green")
actor_scalar = p.add_scalar_bar(mapper=actor.mapper, **sargs)
p.add_axes(color=color)

text = f"{dt.strftime(fmt)}"
actor_text = p.add_text(
    text, position="upper_right", font_size=10, color=color, shadow=False
)

if threshold:
    extra = f"Threshold {threshold:.1e} ({str(plume.units)})"
else:
    extra = "No Threshold"

p.add_text(
    f"Fukushima - {extra}",
    position="upper_left",
    font_size=10,
    color=color,
    shadow=False
)

size, pad = 30, 3
x, y = 10, 10
offset = size * 0.2
p.add_checkbox_button_widget(
    partial(callback_base, actor_base),
    value=True,
    color_on="green",
    color_off="red",
    size=size,
    position=(x, y),
)
p.add_text(
    "Base",
    position=(x + size + offset, y + offset),
    font_size=6,
    color="white",
)

input()

# p.add_slider_widget(callback, (1, n_tsteps), value=1, pointa=(0.6, 0.9), pointb=(0.95, 0.9), slider_width=0.03, title_height=0.02, tube_width=0.003, fmt="%.1f", color=color, title="Time Step", style="modern")
# p.show()
p.add_callback(callback, interval=200)