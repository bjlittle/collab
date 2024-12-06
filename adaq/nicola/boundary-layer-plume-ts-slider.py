"""Execute script with 'python -i <script>'."""

from functools import partial

import geovista
import geovista.theme
from geovista.core import add_texture_coords
from geovista.geodesic import BBox
from geovista.pantry.meshes import regular_grid
from geovista.pantry.data import capitalise
from geovista.qt import GeoBackgroundPlotter
from geovista.raster import wrap_texture

import iris
import netCDF4 as nc
import numpy as np
import pyvista as pv
from pyvista import Actor


show_plume = True
show_boundary = True
show_contour = False
isosurfaces = 200
isosurfaces_range = (None, None)
render = True


# def callback(value) -> None:
def callback_render(value: None) -> None:
    global tstep
    global n_tsteps
    global bl_t
    global bl_data
    global bl_mesh
    global bl_cmap
    global bl_clim
    global p_t
    global p_mesh
    global p_cmap
    global p_clim
    global actor_text
    global actor_scalar
    global actor_plume
    global actor_boundary
    global factor
    global render

    if value is None:
        value = tstep
    else:
        value = int(value)

    tstep = value % n_tsteps
    bl_tstep = tstep // 4

    # if tstep == 0:
        # exit()

    dt = p_t.units.num2date(p_t.points[tstep])
    bl_dt = bl_t.units.num2date(bl_t.points[bl_tstep])

    if render:
        if show_plume:
            data = np.ma.masked_less_equal(p_data[tstep][:], 0).filled(np.nan).flatten()
            p_mesh["data"] = data
            actor_plume = p.add_mesh(p_mesh.threshold(), cmap=p_cmap, clim=p_clim, render=False, reset_camera=False, name="plume",
                       opacity="linear_r", show_scalar_bar=False)
        else:
            actor_plume.SetVisibility(False)

    if show_boundary:
        # bl_mesh["data"] = bl_data[bl_tstep][:].flatten()
        # bl_mesh.save(f"vtk/mesh_boundary_layer_{bl_dt.strftime("%Y%m%d%H%M")}.vtk")
        bl_mesh = pv.read(f"vtk/mesh_boundary_layer_{bl_dt.strftime('%Y%m%d%H%M')}.vtk")
        bl_mesh = bl_mesh.cell_data_to_point_data()
        bl_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True, flip_normals=True)
        bl_mesh.warp_by_scalar(scalars="data", inplace=True, factor=factor)
        # bl_mesh.smooth_taubin(inplace=True, normalize_coordinates=True)
        bl_mesh.active_scalars_name = "data"
        actor_boundary = p.add_mesh(bl_mesh, name="boundary", cmap=bl_cmap, clim=bl_clim, render=False, reset_camera=False, show_scalar_bar=False)
        p.add_actor(actor_scalar)
    else:
        actor_boundary.SetVisibility(False)

    render = True

    text = f"{dt.strftime(fmt)}"
    actor_text.SetText(3, text)


def callback_base(actors: Actor, flag: bool) -> None:
    global render

    render = False
    for actor in actors:
        actor.SetVisibility(flag)
    callback_render(None)


def callback_plume(flag: bool) -> None:
    global show_plume

    show_plume = bool(flag)
    callback_render(None)


def callback_boundary(flag: bool) -> None:
    global show_boundary
    global render

    render = False
    show_boundary = bool(flag)
    callback_render(None)


bl_fname = "data/Boundary_Layer_C1.nc"
bl_cube = iris.load_cube(bl_fname)

run = "R1_T1_V2"
p_fname = f"data/{run}/Fields_grid3_C1.nc"
p_cube = iris.load_cube(p_fname)

# bootstrap
bl_t = bl_cube.coord("time")
bl_y = bl_cube.coord("latitude")
bl_x = bl_cube.coord("longitude")

p_t = p_cube.coord("time")

fmt = "%Y-%m-%d %H:%M"
threshold = None
vmin = threshold if threshold else 50.0
vmax = 640.0  # 639.9715
bl_clim = (vmin, vmax)
bl_cmap = "deep_r"
p_cmap = "fire_r"
p_clim = (0.0, 9e-6)  # R2_T2_V2
p_clim = (0.0, 5.7e-5)  # R1_T1_V1
p_clim = (0.0, 1.6e-5)  # R1_T1_V2
color = "white"
factor = 4e-6  # on-change: regenerate plume.vtk !

bl_ds = nc.Dataset(bl_fname)
bl_data = bl_ds.variables["var__boundary_layer_depth"]

p_ds = nc.Dataset(p_fname)
p_data = p_ds.variables["PM10_AIR_CONCENTRATION"]

p_mesh = pv.read("plume.vtk")

n_tsteps = p_t.shape[0]
tstep = 0

bl_y_cb = bl_y.contiguous_bounds()
bl_x_cb = bl_x.contiguous_bounds()

bl_mesh = geovista.Transform.from_1d(bl_x_cb, bl_y_cb)
bl_mesh["data"] = bl_data[tstep][:].flatten()

data = np.ma.masked_less_equal(p_data[tstep][:], 0).filled(np.nan).flatten()
p_mesh["data"] = data

dt = p_t.units.num2date(p_t.points[tstep])
# bl_mesh.save(f"vtk/mesh_boundary_layer_{dt.strftime("%Y%m%d%H%M")}.vtk")
bl_mesh = pv.read(f"vtk/mesh_boundary_layer_{dt.strftime('%Y%m%d%H%M')}.vtk")
bl_mesh = bl_mesh.cell_data_to_point_data()
bl_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True, flip_normals=True)
bl_mesh.warp_by_scalar(scalars="data", inplace=True, factor=factor)
# bl_mesh.smooth_taubin(inplace=True, normalize_coordinates=True)

bbox = BBox([-1.0, 0.9, 0.9, -1.0], [52.9, 52.9, 51.5, 51.5])    # [52.7, 52.7, 51.7, 51.7])
grid = regular_grid("r1000")
base = bbox.enclosed(grid, preference="point")
base = add_texture_coords(base)
texture = wrap_texture(geovista.natural_earth_hypsometric())

outline = p_mesh.extract_feature_edges()

pv.global_theme.allow_empty_mesh = True
p = GeoBackgroundPlotter()

# p.enable_anti_aliasing(aa_type="ssaa")
p.enable_lightkit()
p.set_background(color="black")

plume = p_mesh.threshold()
actor_plume = p.add_mesh(plume, name="plume", cmap=p_cmap, clim=p_clim, opacity="linear_r", show_scalar_bar=False)

actor_boundary = p.add_mesh(bl_mesh, name="boundary", cmap=bl_cmap, clim=bl_clim, show_edges=False, show_scalar_bar=False)
p.view_poi()

sargs = {"color": color, "title": f"{capitalise(bl_cube.name())} / {str(bl_cube.units)}", "font_family": "arial", "label_font_size": 10, "n_labels": 5}
actor_scalar = p.add_scalar_bar(mapper=actor_boundary.mapper, **sargs)

actor_base = p.add_mesh(base, texture=texture, opacity=0.3, show_edges=True)
p.add_mesh(outline, color="orange", line_width=1)

obs = geovista.Transform.from_points(xs=[0.1313, -0.2408], ys=[52.1951, 52.5703])
actor_obs = p.add_point_labels(obs, ["Cambridge", "Peterborough"], text_color=color, render_points_as_spheres=True, point_color="red", point_size=10, font_size=10, justification_horizontal="right", justification_vertical="top", shape=None)
actor_pts = list(p.actors.values())[-2]

text = f"{dt.strftime(fmt)}"
actor_text = p.add_text(
    text, position="upper_right", font_size=10, color=color, shadow=False
)
p.add_text(
    f"Boundary Layer ({run})",
    position="upper_left",
    font_size=10,
    color=color,
    shadow=False
)
p.add_axes(color=color)

size, pad = 20, 3
x, y = 10, 100
offset = size * 0.2
font_size = 5
p.add_checkbox_button_widget(
    partial(callback_base, [actor_base, actor_obs, actor_pts]),
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
    callback_boundary,
    value=show_boundary,
    color_on="green",
    color_off="red",
    size=size,
    position=(x, y),
)
p.add_text(
    "Boundary Layer",
    position=(x + size + offset, y + offset),
    font_size=font_size,
    color=color,
)

y += size + pad

p.add_slider_widget(callback_render, (0, n_tsteps-1), value=0, pointa=(0.4, 0.9), pointb=(0.95, 0.9), slider_width=0.02, title_height=0.02, tube_width=0.001, fmt="%.0f", color=color, title="Time Step", style="modern")
p.show()
