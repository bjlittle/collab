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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import nc_time_axis
import netCDF4 as nc
import numpy as np
import pyvista as pv
from pyvista import Actor


show_slice = False
show_deposit = True
show_edge = True
show_plume = True
show_smooth = False


# def callback(value) -> None:
def callback() -> None:
    global tstep
    global n_tsteps
    global plume_data
    global plume
    global fmt
    global t
    global actor_text
    global clim
    global p
    global cmap
    global actor_scalar
    global threshold
    global show_slice
    global deposit
    global deposit_data
    global deposit_clim
    global actor_deposit
    global show_deposit
    global actor_plume
    global deposit_null
    global actor_edge
    global show_smooth
    global deposit_cmap

    # value = int(value)

    # tstep = value % n_tsteps
    tstep = (tstep + 1) % n_tsteps

    if tstep == 0:
        # exit()
        tstep = 1
        deposit["data"] = np.zeros(deposit.n_cells)
        deposit["data"] += deposit_data[tstep][:].flatten()

    # print(f"{tstep=}")

    dt = t.units.num2date(t.points[tstep])

    # plume["data"] = np.ma.masked_less_equal(plume_data[tstep][:], 0).filled(np.nan).flatten()
    # plume.active_scalars_name = "data"
    # tmp = plume.threshold()
    # tmp.save(f"vtk/mesh_grid1_{dt.strftime("%Y%m%d%H%M")}.vtk")
    tmp = pv.read(f"vtk/mesh_grid1_{dt.strftime('%Y%m%d%H%M')}.vtk")

    if threshold:
        tmp = tmp.threshold(threshold)

    if show_plume:
        if show_smooth:
            tmp = tmp.extract_geometry().smooth_taubin(n_iter=50, pass_band=0.02, normalize_coordinates=True,
                                                       feature_angle=30, non_manifold_smoothing=True)

        actor_plume = p.add_mesh(tmp, name="plume", cmap=cmap, clim=clim, render=False, reset_camera=False, show_scalar_bar=False, above_color="green")
    else:
        actor_plume.SetVisibility(False)

    p.add_actor(actor_scalar)

    frame = deposit_data[tstep][:].flatten()
    deposit["data"] += frame
    deposit_null["data"] = frame

    if show_deposit:
        actor_deposit = p.add_mesh(deposit.threshold(1e-20), name="deposit", cmap=deposit_cmap, clim=deposit_clim, show_scalar_bar=False)
    else:
        actor_deposit.SetVisibility(False)

    if show_edge:
        actor_edge = p.add_mesh(deposit_null.threshold(1e-20).extract_feature_edges(), name="edge", color="lightgray")
    else:
        actor_edge.SetVisibility(False)

    text = f"{dt.strftime(fmt)}"
    actor_text.SetText(3, text)


def callback_base(actor: Actor, flag: bool) -> None:
    actor.SetVisibility(flag)


def callback_deposit(flag: bool) -> None:
    global show_deposit

    show_deposit = bool(flag)


def callback_edge(flag: bool) -> None:
    global show_edge

    show_edge = bool(flag)


def callback_plume(flag: bool) -> None:
    global show_plume

    show_plume = bool(flag)


def callback_smooth(flag: bool) -> None:
    global show_smooth

    show_smooth = bool(flag)


def callback_threshold(value: float) -> None:
    global threshold

    threshold = value


fname_plume = "fukushima_grid1_201103.nc"
cube_plume = iris.load_cube(fname_plume)

fname_deposit = "fukushima_grid2_201103.nc"

# bootstrap
t = cube_plume.coord("time")
# z = cube_plume.coord("height")
y = cube_plume.coord("latitude")
x = cube_plume.coord("longitude")

fmt = "%Y-%m-%d %H:%M"

# plume_ds = nc.Dataset(fname_plume)
# plume_data = plume_ds.variables["CS137_AIR_CONCENTRATION"]

deposit_ds = nc.Dataset(fname_deposit)
deposit_data = deposit_ds.variables["CS137_DEPOSITION"]

# n_tsteps = t.shape[0]
n_tsteps = 290
tstep = 1

y_cb = y.contiguous_bounds()
x_cb = x.contiguous_bounds()
# z_cb = z.contiguous_bounds()
# print(z_cb)
# z_fix = np.arange(*z_cb.shape) * np.mean(np.diff(y_cb)) * 4
#
# xx, yy, zz = np.meshgrid(x_cb, y_cb, z_fix, indexing="ij")
# shape = xx.shape
# print(f"meshgrid {shape=}")

threshold = None
vmin = threshold if threshold else 0
vmax = 500.0
clim = (vmin, vmax)

cmap = "fire_r"
# cmap = "balance"
color = "white"

# xyz = to_cartesian(xx, yy, zlevel=zz, zscale=0.005)
# print(f"cartesian {xyz.shape=}")
#
# plume = pv.StructuredGrid(xyz[:, 0].reshape(shape), xyz[:, 1].reshape(shape), xyz[:, 2].reshape(shape))
# data = np.ma.masked_less_equal(plume_data[tstep][:], 0).filled(np.nan).flatten()
# plume["idx"] = np.arange(plume.n_cells)
# plume["data"] = data
# plume.set_active_scalars("data", preference="cell")

deposit = geovista.Transform.from_1d(x_cb, y_cb, zlevel=-2)
deposit["data"] = deposit_data[tstep][:].flatten()
deposit_null = deposit.copy()
deposit_clim = (0.0, 7707185.0)  # max
deposit_clim = (0.0, 4000.0)  # mean
deposit_clim = (0.0, 10000.0)
deposit_cmap = "thermal"

pv.global_theme.allow_empty_mesh = True
p = GeoBackgroundPlotter()

# p.enable_anti_aliasing(aa_type="ssaa")
p.enable_lightkit()

bbox = BBox(lons=[132.0, 149.4, 149.4, 132.0], lats=[46.5, 46.5, 30.0, 30.0])
coasts = bbox.enclosed(coastlines(resolution="10m", zlevel=0))
base = add_texture_coords(bbox.enclosed(regular_grid("r400"), preference="center"))
base_edge = base.extract_feature_edges()
texture = wrap_texture(geovista.natural_earth_hypsometric())

p.set_background(color="black")

sargs = {"color": color, "title": f"{capitalise(cube_plume.name())} \n {str(cube_plume.units)}", "font_family": "arial", "label_font_size": 10, "n_labels": 5}

actor_base = p.add_mesh(base, texture=texture, opacity=0.3, show_edges=True)
p.view_poi()
p.add_mesh(coasts, color="gray", line_width=1)
p.add_mesh(base_edge, color="gray", line_width=1)
# p.add_mesh(bbox.boundary(), color="orange", line_width=2)

obs = geovista.Transform.from_points(xs=[141.03305556], ys=[37.42305556])
p.add_point_labels(obs, ["Fukushima\nDaiichi   "], render_points_as_spheres=True, point_color="red", point_size=10, font_size=10, text_color="white", justification_horizontal="right", justification_vertical="top", shape=None)

# outline = plume.extract_feature_edges()
# outline.save(f"vtk/outline.vtk")
outline = pv.read(f"vtk/outline.vtk")
p.add_mesh(outline, color="orange", line_width=1)

dt = t.units.num2date(t.points[tstep])
# tmp = plume.threshold()
# tmp.save(f"vtk/mesh_grid1_{dt.strftime("%Y%m%d%H%M")}.vtk")
ptmp = pv.read(f"vtk/mesh_grid1_{dt.strftime('%Y%m%d%H%M')}.vtk")

if threshold:
    ptmp = ptmp.threshold(threshold)

if show_smooth:
    ptmp = ptmp.extract_geometry().smooth_taubin(n_iter=50, pass_band=0.02, normalize_coordinates=True, feature_angle=30, non_manifold_smoothing=True)

actor_plume = p.add_mesh(ptmp, scalars="data", name="plume", cmap=cmap, clim=clim, show_scalar_bar=False, above_color="green")
actor_scalar = p.add_scalar_bar(mapper=actor_plume.mapper, **sargs)
p.add_axes(color=color)

dtmp = deposit.threshold(1e-20)
actor_deposit = p.add_mesh(dtmp, name="deposit", cmap=deposit_cmap, clim=deposit_clim, show_scalar_bar=False)
actor_edge = p.add_mesh(dtmp.extract_feature_edges(), name="edge", color="lightgray")

text = f"{dt.strftime(fmt)}"
actor_text = p.add_text(
    text, position="upper_right", font_size=10, color=color, shadow=False
)

p.add_text(
    f"Fukushima Daiichi Nuclear Power Plant",
    position="upper_left",
    font_size=10,
    color=color,
    shadow=False
)

size, pad = 20, 3
x, y = 10, 100
offset = size * 0.2
font_size = 5
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
    font_size=font_size,
    color=color,
)

y += size + pad

p.add_checkbox_button_widget(
    callback_deposit,
    value=show_deposit,
    color_on="green",
    color_off="red",
    size=size,
    position=(x, y),
)
p.add_text(
    "Deposition (Cumulative)",
    position=(x + size + offset, y + offset),
    font_size=font_size,
    color=color,
)

y += size + pad

p.add_checkbox_button_widget(
    callback_edge,
    value=show_edge,
    color_on="green",
    color_off="red",
    size=size,
    position=(x, y),
)
p.add_text(
    "Deposition (Instantaneous)",
    position=(x + size + offset, y + offset),
    font_size=font_size,
    color=color,
)

y += size + pad

p.add_checkbox_button_widget(
    callback_plume,
    value=show_plume,
    color_on="green",
    color_off="red",
    size=size,
    position=(x, y),
)
p.add_text(
    "Plume",
    position=(x + size + offset, y + offset),
    font_size=font_size,
    color=color,
)

y += size + pad

p.add_checkbox_button_widget(
    callback_smooth,
    value=show_smooth,
    color_on="green",
    color_off="red",
    size=size,
    position=(x, y),
)
p.add_text(
    "Smooth",
    position=(x + size + offset, y + offset),
    font_size=font_size,
    color=color,
)


input()


p.add_slider_widget(callback_threshold, (0, 10), value=0, pointa=(0.4, 0.9), pointb=(0.95, 0.9), slider_width=0.02, title_height=0.02, tube_width=0.001, fmt="%.4f", color=color, title=f"Threshold ({str(cube_plume.units)})", style="modern")
# p.show()
p.add_callback(callback, interval=200)