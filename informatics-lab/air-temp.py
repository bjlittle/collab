from pathlib import Path
from typing import List, Optional

import cartopy.io.shapereader as shp
import cf_units
import geovista
import geovista.theme
import netCDF4 as nc
import numpy as np
from pyproj import CRS, Transformer
import pyvista as pv
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon


def clean(names):
    single = False
    if isinstance(names, str):
        names = [names]
        single = True
    cleaned = []
    for parts in names:
        parts = parts.replace("_", " ")
        name = " ".join([part.capitalize() for part in parts.split()])
        cleaned.append(name)
    if single:
        result = cleaned[0]
    else:
        result = np.array(cleaned)
    return result


# TODO: replace when available from geovista release
def coastline_geometries(
    fname: str,
    closed: Optional[bool] = True,
) -> List[np.ndarray]:
    lines, multi_lines = [], []

    # load in the shapefiles
    reader = shp.Reader(fname)

    def unpack(geometries):
        for geometry in geometries:
            if isinstance(geometry, (MultiLineString, MultiPolygon)):
                multi_lines.extend(list(geometry.geoms))
            else:
                coords = (
                    geometry.exterior.coords
                    if isinstance(geometry, Polygon)
                    else geometry.coords
                )
                data = coords[:-1] if not closed and geometry.is_closed else coords[:]
                xy = np.array(data, dtype=np.float32)
                x = xy[:, 0].reshape(-1, 1)
                y = xy[:, 1].reshape(-1, 1)
                z = np.zeros_like(x)
                xyz = np.hstack((x, y, z))
                lines.append(xyz)

    unpack(reader.geometries())
    if multi_lines:
        unpack(multi_lines)

    return lines


# TODO: replace when available from geovista release
def coastline_mesh_line(
    fname: str,
) -> pv.PolyData:
    geoms = coastline_geometries(fname)
    npoints_per_geom = [geom.shape[0] for geom in geoms]
    ngeoms = len(geoms)
    geoms = np.concatenate(geoms)
    nlines = geoms.shape[0] - ngeoms

    # convert geometries to a vtk line mesh
    mesh = pv.PolyData()
    mesh.points = geoms
    lines = np.full((nlines, 3), 2, dtype=np.int32)
    pstart, lstart = 0, 0

    for npoints in npoints_per_geom:
        pend = pstart + npoints
        lend = lstart + npoints - 1
        lines[lstart:lend, 1] = np.arange(pstart, pend - 1, dtype=np.int32)
        lines[lstart:lend, 2] = np.arange(pstart + 1, pend, dtype=np.int32)
        pstart, lstart = pend, lend

    mesh.lines = lines

    fname = Path(fname)
    prj = fname.parent / f"{fname.stem}.prj"
    if prj.is_file():
        with open(prj, mode="r") as fh:
            defn = fh.readlines()
        crs = CRS.from_user_input(defn[0])
        #wkt = np.array([crs.to_wkt()])
        #mesh.field_data["gvCRS"] = wkt
        transformer = Transformer.from_crs(crs, geovista.crs.WGS84, always_xy=True)
        lons, lats = transformer.transform(mesh.points[:, 0], mesh.points[:, 1])
        result = geovista.common.to_xyz(lons, lats)
        mesh.points = result

    return mesh


def callback():
    global mesh
    global step
    global nsteps
    global data
    global units
    global time
    global actor

    step = (step + 24) % nsteps
    sdata = np.ravel(data[step])
    mesh.cell_data["data"] = sdata
    actor.SetText(3, str(units.num2date(time[step])))


fname = "20181231T0700Z-20200101T0600Z-air_temperature_at_screen_level.nc"
ds = nc.Dataset(fname)

lon = ds.variables["longitude_bnds"][:]
lat = ds.variables["latitude_bnds"][:]
time = ds.variables["time"]
units = cf_units.Unit(time.units)
data = ds.variables["air_temperature_at_screen_level"]

fname1 = "LSOA_2004_London_Low_Resolution.shp"
fname2 = "LSOA_2011_London_gen_MHW.shp"
fname3 = "London_Borough_Excluding_MHW.shp"
fname4 = "London_Ward.shp"
fname5 = "London_Ward_CityMerged.shp"
fname6 = "MSOA_2004_London_High_Resolution.shp"
fname7 = "MSOA_2011_London_gen_MHW.shp"
fname8 = "OA_2011_London_gen_MHW.shp"
fname = Path(f"statistical-gis-boundaries-london/ESRI/{fname1}")
line = coastline_mesh_line(str(fname))

interval = 250
nsteps = data.shape[0]
step = 5
tstep = units.num2date(time[step])

mesh = geovista.Transform.from_1d(
    lon,
    lat,
    data=data[step],
    name="data",
)

clim = (270, 300)
plotter = geovista.GeoBackgroundPlotter()

line.field_data["gvCRS"] = mesh.field_data["gvCRS"]

title = f"{clean(data.long_name)} / {data.units}"
sargs = dict(title=title, shadow=True)

plotter.add_mesh(line, color="black", line_width=2)
plotter.add_mesh(mesh, cmap="balance", clim=clim, above_color="green",
                 below_color="yellow", show_edges=False, lighting=False,
                 scalar_bar_args=sargs)

plotter.add_axes()
plotter.scalar_bar.GetAnnotationTextProperty().SetFontSize(5)

actor = plotter.add_text(
    str(tstep), position="upper_right", font_size=10, shadow=True
)

plotter.add_callback(callback, interval=interval)