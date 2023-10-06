"""Execute script with 'python -i <script>'."""
from pathlib import Path

from cf_units import Unit
import cmocean
import geovista
from geovista.transform import transform_mesh
from geovista.qt import GeoBackgroundPlotter
import netCDF4 as nc
import numpy as np


def callback() -> None:
    global step
    global n_steps
    global fnames
    global fname
    global ds
    global data
    global mesh
    global scalars
    global fmt
    global time
    global unit
    global actor

    step = (step + 1) % n_steps

    step_fname = fnames[step // 24]

    if step_fname != fname:
        ds = nc.Dataset(step_fname)
        data = ds.variables["toa_outgoing_longwave_flux"]
        time = ds.variables["time"]
        fname = step_fname

    t = step % 24
    idxs = mesh["idxs"]
    mesh[scalars] = np.ravel(data[t][:])[idxs]
    mesh.active_scalars_name = scalars
    actor.SetText(3, unit.num2date(time[t]).strftime(fmt))


# sort the assets in date ascending date order
path = Path("./assets/global")
fnames = list(path.glob("**/*.nc"))
fnames_by_dt = [(int(fname.name.split("_")[0]), fname) for fname in fnames]
fnames = [fname for _, fname in sorted(fnames_by_dt, key=lambda pair: pair[0])]

# bootstrap
n_fnames = len(fnames)
n_steps = n_fnames * 24
step = 0
fname = fnames[0]

# load the first asset
ds = nc.Dataset(fname)
lons = ds.variables["longitude_bnds"][:]
lats = ds.variables["latitude_bnds"][:]
time = ds.variables["time"]
data = ds.variables["toa_outgoing_longwave_flux"]

unit = Unit(time.units)
fmt = "%Y-%m-%d %H:%M"

# create the mesh
scalars = "data"
mesh = geovista.Transform.from_1d(lons, lats, data=data[step][:], name=scalars)

mesh["idxs"] = np.arange(mesh.n_faces)
tgt_crs = "+proj=robin"
mesh = transform_mesh(mesh, tgt_crs=tgt_crs)

cmap = cmocean.tools.crop_by_percent(cmocean.cm.gray_r, 20, which="max")
opacity = [1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0]
clim = (69.45703, 416.0235)

plotter = GeoBackgroundPlotter(crs=tgt_crs)
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

plotter.view_xy()
plotter.camera.zoom(1.6)

plotter.add_callback(callback, interval=100)
