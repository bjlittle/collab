"""Execute script with 'python -i <script>'."""
from pathlib import Path

from cf_units import Unit
import geovista.theme
from geovista.common import to_cartesian
from geovista.pantry.data import capitalise
from geovista.qt import GeoBackgroundPlotter
import iris
import matplotlib as mpl
import netCDF4 as nc
import numpy as np
import pyvista as pv


def rgb(r, g, b):
    return (r / 256, g / 256, b / 256, 1.0)


def rgba(r, g, b):
    return np.array([r / 256, g / 256, b / 256, 1.0])


def daqi(vmin, vmax):
    from matplotlib.colors import ListedColormap

    # https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info&pollutant=so2#pollutant

    mapping = np.linspace(vmin, vmax, 256)
    colors = np.empty((256, 4))

    c01 = rgba(156, 255, 156)   # 0-88 low
    c02 = rgba(49, 255, 0)      # 89-177 low
    c03 = rgba(49, 207, 0)      # 178-266 low
    c04 = rgba(255, 255, 0)     # 267-354 moderate
    c05 = rgba(255, 207, 0)     # 355-443 moderate
    c06 = rgba(255, 154, 0)     # 444-532 moderate
    c07 = rgba(255, 100, 100)   # 533-710 high
    c08 = rgba(255, 0, 0)       # 711-887 high
    c09 = rgba(153, 0, 0)       # 888-1064 high
    c10 = rgba(206, 48, 255)    # 1065- high

    # colors[mapping >= 1065] = c10
    colors[mapping < 1065] = c09
    colors[mapping < 888] = c08
    colors[mapping < 711] = c07
    colors[mapping < 533] = c06
    colors[mapping < 444] = c05
    colors[mapping < 355] = c04
    colors[mapping < 267] = c03
    colors[mapping < 178] = c02
    colors[mapping < 89] = c01

    return ListedColormap(colors, N=256)


def callback() -> None:
    global step
    global n_steps
    global data
    global mesh
    global fmt
    global t
    global unit
    global actor
    global clim
    global p
    global cmap
    global sargs
    global show_edges

    step = (step + 1) % n_steps

    if step == 0:
        step = 7

    # tdata = np.ma.masked_less_equal(data[step][:], 0).filled(np.nan).flatten() * 1000000
    # print(f"{step=}, {np.nanmean(tdata)=}, {np.nanmax(tdata)=}")
    # mesh["data"] = tdata
    tmp = pv.read(f"vtk/mesh_{step}.vtk").extract_geometry().smooth_taubin(n_iter=50, pass_band=0.05)
    print(f"{step=}")
    # mesh.active_scalars_name = "data"
    # tmp = mesh.threshold(25)
    # tmp.save(f"vtk/mesh_{step}.vtk")
    p.add_mesh(tmp, name="plume", cmap=cmap, clim=clim, render=False, reset_camera=False, scalar_bar_args=sargs, show_edges=show_edges, above_color=rgb(206, 48, 255))
    actor.SetText(3, unit.num2date(t.points[step]).strftime(fmt))


# sort the assets in date ascending date order
fname = "so2-air-concentration-timeseries.nc"
cube = iris.load_cube(fname)

ds = nc.Dataset(fname)
data = ds.variables["SULPHUR_DIOXIDE_AIR_CONCENTRATION"]

# bootstrap
t = cube.coord("time")
z = cube.coord("altitude")
y = cube.coord("latitude")
x = cube.coord("longitude")

unit = Unit(t.units)
fmt = "%Y-%m-%d %H:%M"

n_steps = t.shape[0]
n_steps = 172
step = 7

y_cb = y.contiguous_bounds()
x_cb = x.contiguous_bounds()
z_cb = z.contiguous_bounds()
print(z_cb)
z_fix = np.arange(*z_cb.shape) * np.mean(np.diff(y_cb)) * 3

xx, yy, zz = np.meshgrid(x_cb, y_cb, z_fix, indexing="ij")
shape = xx.shape
print(f"meshgrid {shape=}")

dmin, dmax = 0.0, 0.043242946 * 1000000
dmin, dmax = 25.0, 1065.0
clim = (dmin, dmax)
# for i in np.arange(tmax):
#     data = cube[i].data
#     min, max = np.min(data), np.max(data)
#     print(f"{min=}, {max=}")
#     if min < dmin:
#         dmin = min
#     if max > dmax:
#         dmax = max
# print(f"{dmin=}, {dmax=}")

xyz = to_cartesian(xx, yy, zlevel=zz, zscale=0.005)
print(f"cartesian {xyz.shape=}")

mesh = pv.StructuredGrid(xyz[:, 0].reshape(shape), xyz[:, 1].reshape(shape), xyz[:, 2].reshape(shape))
tdata = np.ma.masked_less_equal(data[step][:], 0).filled(np.nan).flatten() * 1000000
print(f"{step=}, {np.nanmean(tdata)=}, {np.nanmax(tdata)=}")
mesh["data"] = tdata

outline = mesh.extract_feature_edges()

cmap = mpl.colormaps.get_cmap("cet_CET_L17").resampled(lutsize=9)
cmap = "magma_r"
#cmap = daqi(dmin, dmax)
show_edges = True

pv.global_theme.allow_empty_mesh = True
p = GeoBackgroundPlotter()

p.add_points(xs=(-22.55499778,), ys=(63.868663192,), render_points_as_spheres=True, color="red", point_size=10)

sargs = {"color": "white", "title": f"{capitalise(cube.name())} (µ{str(cube.units)})", "title_font_size": 15, "label_font_size": 10, "fmt": "%.1f"}

p.set_background(color="black")
p.add_base_layer(texture=geovista.blue_marble(), zlevel=0, resolution="c192")
# p.add_mesh(outline, line_width=2, color="gray")
mesh.active_scalars_name = "data"
# tmp = mesh.threshold(25)
# tmp.save(f"vtk/mesh_{step}.vtk")
tmp = pv.read(f"vtk/mesh_{step}.vtk").extract_geometry().smooth_taubin(n_iter=50, pass_band=0.05)
p.add_mesh(tmp, name="plume", cmap=cmap, clim=clim, scalar_bar_args=sargs, show_edges=show_edges, above_color=rgb(206, 48, 255))
p.add_coastlines()
p.add_axes(color="white")

p.add_text(
    "Reykjanes, May 2024", position="upper_left", font_size=10, color="white", shadow=False
)

text = unit.num2date(t.points[step]).strftime(fmt)
actor = p.add_text(
    text, position="upper_right", font_size=10, color="white", shadow=False
)

p.add_callback(callback, interval=200)
