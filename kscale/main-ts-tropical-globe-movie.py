from pathlib import Path

from cf_units import Unit
import cmocean
import geovista
import iris
import netCDF4 as nc
import numpy as np


# sort the assets in date ascending date order
path = Path("./assets/native")
fnames = list(path.glob("**/*.nc"))
fnames_by_dt = [(int(fname.name.split("_")[0]), fname) for fname in fnames]
fnames = [fname for _, fname in sorted(fnames_by_dt, key=lambda pair: pair[0])]

# bootstrap
n_fnames = len(fnames)
n_steps = n_fnames * 24
step = 0
fname = fnames[0]

cube = iris.load_cube(fname)
lons = cube.coord("longitude")
lats = cube.coord("latitude")
lons = lons.contiguous_bounds()
lats = lats.contiguous_bounds()
del cube

# load the first asset
ds = nc.Dataset(fname)
# lons = ds.variables["longitude_bnds"][:]
# lats = ds.variables["latitude_bnds"][:]
time = ds.variables["time"]
data = ds.variables["toa_outgoing_longwave_flux"]

unit = Unit(time.units)
fmt = "%Y-%m-%d %H:%M"

# create the mesh
scalars = "data"
mesh = geovista.Transform.from_1d(lons, lats, data=data[step][:], name=scalars)

cmap = cmocean.tools.crop_by_percent(cmocean.cm.gray_r, 20, which="max")
opacity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
clim = (68.71875, 418.42188)

plotter = geovista.GeoPlotter(off_screen=True)

plotter.open_movie("globe.mp4")

plotter.add_mesh(
    mesh, cmap=cmap, clim=clim, opacity=opacity, show_scalar_bar=False, lighting=False
)

plotter.add_coastlines(color="black")
plotter.set_background(color="black")
plotter.add_base_layer(texture=geovista.natural_earth_1())

text = unit.num2date(time[step]).strftime(fmt)
actor = plotter.add_text(
    text, position="upper_right", font_size=10, color="white", shadow=False
)

plotter.add_axes(color="white")
plotter.view_xz(negative=True)
plotter.camera.zoom(2.0)

plotter.show(auto_close=False)

plotter.write_frame()

for step in range(n_steps):
    print(f"{step=} [{n_steps}]")
    step_fname = fnames[step // 24]

    if step_fname != fname:
        ds = nc.Dataset(step_fname)
        data = ds.variables["toa_outgoing_longwave_flux"]
        time = ds.variables["time"]
        fname = step_fname

    t = step % 24
    mesh[scalars] = np.ravel(data[t][:])
    mesh.active_scalars_name = scalars
    actor.SetText(3, unit.num2date(time[t]).strftime(fmt))
    plotter.write_frame()

plotter.close()
