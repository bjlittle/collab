"""Execute script with 'python -i <script>'."""
from datetime import datetime, timedelta
from pathlib import Path

from cf_units import Unit
import geovista
import geovista.theme
from geovista.common import to_cartesian
from geovista.pantry.data import capitalise
from geovista.qt import GeoBackgroundPlotter
import iris
import matplotlib as mpl
import netCDF4 as nc
import numpy as np
import pyvista as pv
from geovista.geodesic import line


def callback(value) -> None:
    global forecasts
    global tstep
    global n_tsteps
    global pstep
    global n_psteps
    global payload
    global mesh
    global fmt
    global t
    global unit
    global actor
    global clim
    global p
    global cmap
    global sargs

    value = int(value)

    tstep = value % n_tsteps
    pstep = value // n_tsteps

    print(f"{pstep=}, {tstep=}")

    # mesh["data"] = np.ma.masked_less_equal(payload[pstep][tstep][:], 0).filled(np.nan).flatten()
    # mesh.active_scalars_name = "data"
    # tmp = mesh.threshold()
    # tmp.save(f"vtk/mesh_forecast_{forecasts[pstep]}_{tstep}.vtk")
    p.subplot(0, 0)
    tmp = pv.read(f"vtk/mesh_apriori_{forecasts[pstep]}_{tstep}.vtk").threshold(0.2)  #.extract_geometry().smooth_taubin(n_iter=50, pass_band=0.05)
    p.add_mesh(tmp, name="apriori", cmap=cmap, clim=clim, render=False, reset_camera=False, scalar_bar_args=sargs, show_edges=True)

    p.subplot(0, 1)
    tmp = pv.read(f"vtk/mesh_forecast_{forecasts[pstep]}_{tstep}.vtk").threshold(0.2)  # .extract_geometry().smooth_taubin(n_iter=50, pass_band=0.05)
    p.add_mesh(tmp, name="forecast", cmap=cmap, clim=clim, render=False, reset_camera=False, scalar_bar_args=sargs, show_edges=True)

    f = forecasts[pstep]
    ft = datetime(int(f[:4]), int(f[4:6]), int(f[6:8]), int(f[8:10]), int(f[10:12]))
    td = timedelta((delta := 3 * tstep) / 24)
    text = f"{ft.strftime(fmt)} T+{delta:02d} ({(ft + td).strftime(fmt)})"
    actor.SetText(0, text)

    # if pstep == 0 and tstep == 0:
    #     exit()


forecasts = [
#    "202111030800",
    "202111030900",
    "202111031200",
    "202111031800",
    "202111040000",
    "202111040600",
    "202111041200",
    "202111041800",
    "202111050000",
    "202111050600",
    "202111051200",
    "202111051800",
]
payload = []

# for forecast in forecasts:
#     fname = f"data/karymsky_forecast_{forecast}.nc"
#     ds = nc.Dataset(fname)
#     payload.append(ds.variables["volcanic_ash_air_concentration"])

fname = f"data/karymsky_apriori_202111030800.nc"
cube = iris.load_cube(fname)

# bootstrap
t = cube.coord("time")
z = cube.coord("flight_level")
y = cube.coord("latitude")
x = cube.coord("longitude")

unit = Unit(t.units)
fmt = "%Y-%m-%d %H:%M"

n_tsteps = t.shape[0]
tstep = 0
n_psteps = len(forecasts)
pstep = 0

# y_cb = y.contiguous_bounds()
# x_cb = x.contiguous_bounds()
# z_cb = z.contiguous_bounds()
# print(z_cb)
# z_fix = np.arange(*z_cb.shape) * np.mean(np.diff(y_cb)) * 3
#
# xx, yy, zz = np.meshgrid(x_cb, y_cb, z_fix, indexing="ij")
# shape = xx.shape
# print(f"meshgrid {shape=}")

vmin, vmax = 0.2, 64.75421
clim = (vmin, vmax)
cmap = "fire_r"

# xyz = to_cartesian(xx, yy, zlevel=zz, zscale=0.005)
# print(f"cartesian {xyz.shape=}")
#
# mesh = pv.StructuredGrid(xyz[:, 0].reshape(shape), xyz[:, 1].reshape(shape), xyz[:, 2].reshape(shape))
# data = np.ma.masked_less_equal(payload[pstep][tstep][:], 0).filled(np.nan).flatten()
# mesh["data"] = data

pv.global_theme.allow_empty_mesh = True
p = GeoBackgroundPlotter(shape=(1, 2))

sargs = {"color": "white", "title": f"{capitalise(cube.name())} / {str(cube.units)}"}

p.subplot(0, 0)
p.set_background(color="black")
p.add_base_layer(texture=geovista.blue_marble(), zlevel=0)
p.add_points(xs=(159.442222,), ys=(54.047778,), render_points_as_spheres=True, color="red", point_size=10)
p.add_mesh(line(-180, [90, 0, -90]), color="orange", line_width=3)
# tmp = mesh.threshold()
# tmp.save(f"vtk/mesh_forecast_{forecasts[pstep]}_{tstep}.vtk")
tmp = pv.read(f"vtk/mesh_apriori_{forecasts[pstep]}_{tstep}.vtk").threshold(0.2)  #.extract_geometry().smooth_taubin(n_iter=50, pass_band=0.05)
p.add_mesh(tmp, scalars="data", name="apriori", cmap=cmap, clim=clim, scalar_bar_args=sargs, show_edges=False)
p.add_coastlines()
p.add_axes(color="white")

p.link_views()

f = forecasts[pstep]
ft = datetime(int(f[:4]), int(f[4:6]), int(f[6:8]), int(f[8:10]), int(f[10:12]))
td = timedelta((delta := 3 * tstep) / 24)

text = f"{ft.strftime(fmt)} T+{delta:02d} ({(ft + td).strftime(fmt)})"
actor = p.add_text(
    text, position="lower_left", font_size=10, color="white", shadow=False
)
p.add_text("Karymsky: a priori", position="upper_left", font_size=10, color="white", shadow=False)

p.subplot(0, 1)
p.add_base_layer(texture=geovista.blue_marble(), zlevel=0)
p.add_points(xs=(159.442222,), ys=(54.047778,), render_points_as_spheres=True, color="red", point_size=10)
p.add_mesh(line(-180, [90, 0, -90]), color="orange", line_width=3)
# tmp = mesh.threshold()
# tmp.save(f"vtk/mesh_forecast_{forecasts[pstep]}_{tstep}.vtk")
tmp = pv.read(f"vtk/mesh_forecast_{forecasts[pstep]}_{tstep}.vtk").threshold(0.2)  #.extract_geometry().smooth_taubin(n_iter=50, pass_band=0.05)
p.add_mesh(tmp, scalars="data", name="forecast", cmap=cmap, clim=clim, scalar_bar_args=sargs, show_edges=False)
p.add_coastlines()
# p.add_axes(color="white")
p.add_text("Karymsky: forecast", position="upper_left", font_size=10, color="white", shadow=False)

p.add_slider_widget(callback, (0, n_psteps*n_tsteps), value=0, color="white", fmt="%.0f")
p.show()
