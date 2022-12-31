from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cartopy.io.shapereader as shp
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import netCDF4 as nc
import numpy as np
import pyvista as pv
from pyvista import _vtk
from pyvista.core.filters import _get_output
from pyproj import CRS, Transformer
from pyvistaqt import BackgroundPlotter
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

import geovista.theme


# typing aliases
CellIDs = List[int]
CellIDLike = Union[int, CellIDs]
LIMIT = Tuple[float, float]
PathLike = Union[Path, str]
XYZ = Tuple[float, float, float]

#: Geographic WGS84
WGS84 = CRS.from_user_input("epsg:4326")

#: Extrude factor
EXTRUDE: int = 2000
EXTRUDE_BASE: int = 250


@dataclass
class Points:
    x: float
    y: float
    xyz: XYZ


@dataclass
class Frame:
    nobs: int
    index: np.ndarray  # master index into data for each site
    cids: np.ndarray  # parent mesh cell (polygon) index for each site
    sites: np.ndarray  # site names, sorted alphabetically (unique)
    pois: np.ndarray  # site coordinates in native CRS
    neighbours: dict  # the sites sharing which parent cell
    dt: np.ndarray  # time-series datetime strings
    data: np.ndarray  # contiguous time-series payload for each site
    clim: LIMIT  # data payload minimum and maximum
    extruded: pv.PolyData  # extruded site regions


def clean(names):
    single = False
    if isinstance(names, str):
        names = [names]
        single = True
    cleaned = []
    for parts in names:
        parts = parts.replace("_", " ")
        name = " ".join([part.capitalize() for part in parts.split()])
        # >=py310 syntax
        match name:
            case "Fine Offset 1":
                name = "Fine Offset"
            case "Kt199hz":
                name = "KT199HZ"
            case "PagesHillAllotment" | "Pageshillallotment":
                name = "Pages Hill Allotment"
            case "Unknown 14":
                name = "Unknown 1"
            case "Unknown 25":
                name = "Unknown 2"
            case "Unknown 59" | "Unknown 4":
                name = "Unknown 3"
            case "WeatherN15" | "Weathern15":
                name = "Weather N15"
        cleaned.append(name)
    if single:
        result = cleaned[0]
    else:
        result = np.array(cleaned)
    return result


def prepare(fname_shape: PathLike, fname_density: PathLike, mesh: pv.PolyData) -> Frame:
    ds = nc.Dataset(fname_shape)
    sitename = clean(ds.variables["SiteName"][:])
    data = ds.variables["Temperature_K"][:]
    datetime = ds.variables["DateTime"][:]
    lat = ds.variables["Lat"][:]
    lon = ds.variables["Lon"][:]
    ds.close()

    ds = nc.Dataset(fname_density)
    dsitename = clean(ds.variables["SiteName"][:])
    dvalue = ds.variables["bldg_density"][:]
    ds.close()

    sites, index = np.unique(sitename, return_index=True)
    assert set(sites) == set(dsitename)
    dt = np.unique(datetime)
    nobs = dt.size
    ilat, ilon = lat[index], lon[index]

    crs = CRS.from_wkt(str(mesh.field_data["crs"][0]))
    transformer = Transformer.from_crs(WGS84, crs, always_xy=True)
    xs, ys = transformer.transform(ilon, ilat, errcheck=False)

    xisfinite, yisfinite = np.isfinite(xs), np.isfinite(ys)
    assert np.array_equal(xisfinite, yisfinite)

    cids = [-1] * index.size
    for i, (isfinite, x, y) in enumerate(zip(xisfinite, xs, ys)):
        if isfinite:
            cid = find_nearest_cell(mesh, x, y)
            cids[i] = cid[0]
    cids = np.array(cids)

    ucids, counts = np.unique(cids, return_counts=True)

    multiple = np.where(counts > 1)[0]
    neighbours = dict()
    for i in multiple:
        cid = ucids[i]
        neighbours[cid] = np.where(cids == cid)[0]

    clim = np.floor(data.min()), np.ceil(data.max())

    density = np.empty(sites.shape, dtype=float)
    for name, value in zip(dsitename, dvalue):
        i = np.where(sites == name)[0]
        density[i] = value

    for i in neighbours.values():
        density[i] = np.mean(density[i])
    max_density = density.max()

    zs = (density / max_density) * EXTRUDE
    shape = (-1, 1)
    pois = np.hstack([xs.reshape(shape), ys.reshape(shape), zs.reshape(shape)])

    extruded = {}
    for cid in ucids:
        if cid >= 0:
            region = cast_UnstructuredGrid_to_PolyData(
                mesh.extract_cells(cid)
            ).triangulate()
            if cid in neighbours:
                i = neighbours[cid][0]
            else:
                i = np.where(cids == cid)[0]
            extrude = (density[i] / max_density) * EXTRUDE
            region.extrude((0, 0, EXTRUDE_BASE + extrude), capping=True, inplace=True)
            region.translate((0, 0, -EXTRUDE_BASE), inplace=True)
            site_index = np.where(cids == cid)[0][0]
            extruded[site_index] = region

    mesh.remove_cells(ucids, inplace=True)
    mesh.set_active_scalars("data", preference="cell")

    frame = Frame(
        nobs=nobs,
        index=index,
        cids=cids,
        sites=sites,
        pois=pois,
        neighbours=neighbours,
        dt=dt,
        data=data,
        clim=clim,
        extruded=extruded,
    )

    return frame


def find_nearest_cell(
    mesh: pv.PolyData,
    x: float,
    y: float,
    z: Optional[float] = 0,
    single: Optional[bool] = False,
) -> CellIDLike:
    poi = np.array([x, y, z])
    cid = mesh.find_closest_cell(poi)

    pids = np.asanyarray(mesh.cell_point_ids(cid))
    points = mesh.points[pids]
    mask = np.all(np.isclose(points, poi), axis=1)
    poi_is_vertex = np.any(mask)

    if poi_is_vertex:
        pid = pids[mask][0]
        result = sorted(mesh.extract_points(pid)["vtkOriginalCellIds"])
    else:
        result = [cid]

    if single:
        if (n := len(result)) > 1:
            emsg = f"Expected to find 1 cell but found {n}, " f"got CellIDs {result}."
            raise ValueError(emsg)
        (result,) = result

    return result


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

    return mesh


def coastline_mesh(
    fname: str,
) -> pv.PolyData:
    geoms = coastline_geometries(fname, closed=False)
    npoints_per_geom = [geom.shape[0] for geom in geoms]
    cumsum = np.cumsum(npoints_per_geom)
    ngeoms = len(geoms)
    vertices = np.concatenate(geoms)
    start = 0
    faces = []
    for i, end in enumerate(cumsum):
        faces.append(np.concatenate([[npoints_per_geom[i]], np.arange(start, end)]))
        start = end
    faces = np.concatenate(faces)
    mesh = pv.PolyData(vertices, faces, n_faces=ngeoms)

    fname = Path(fname)
    prj = fname.parent / f"{fname.stem}.prj"
    if prj.is_file():
        with open(prj, mode="r") as fh:
            defn = fh.readlines()
        crs = CRS.from_user_input(defn[0])
        wkt = np.array([crs.to_wkt()])
        mesh.field_data["crs"] = wkt

    return mesh


def cast_UnstructuredGrid_to_PolyData(
    mesh: pv.UnstructuredGrid,
    clean: Optional[bool] = False,
) -> pv.PolyData:
    """
    TODO

    Notes
    -----
    .. versionadded:: 0.1.0

    """
    if not isinstance(mesh, pv.UnstructuredGrid):
        dtype = type(mesh).split(" ")[1][:-1]
        emsg = f"Expected a 'pyvista.UnstructuredGrid', got {dtype}."
        raise TypeError(emsg)

    # see https://vtk.org/pipermail/vtkusers/2011-March/066506.html
    alg = _vtk.vtkGeometryFilter()
    alg.AddInputData(mesh)
    alg.Update()
    result = _get_output(alg)

    if clean:
        result = result.clean()

    return result


def callback() -> None:
    global previous_step
    global step
    global data
    global frame
    global dt_actor
    global plotter
    global ax
    global line
    global chart_x
    global chart_y

    obs = frame.data[frame.index + step].filled(np.nan)
    average = np.nanmean(obs)
    for cid, sites in frame.neighbours.items():
        cell_mean = np.mean(obs[sites])
        obs[sites] = cell_mean
    data[frame.cids] = obs

    dt_actor.SetText(3, frame.dt[step])

    for site, region in frame.extruded.items():
        region.cell_data["data"] = obs[site]
        region.cell_data["data"] = obs[site]
        region.set_active_scalars("data", preference="cell")

    if line is None:
        chart_x = [datetime.strptime(frame.dt[step], "%Y-%m-%d %H:%M:%S")]
        chart_y = [average]
        line = ax.plot(chart_x, chart_y, color="cyan", linewidth=0.5)[0]
    else:
        if previous_step > step:
            chart_x, chart_y = [], []
        x = datetime.strptime(frame.dt[step], "%Y-%m-%d %H:%M:%S")
        chart_x.append(x)
        chart_y.append(average)
        line.set_xdata(chart_x)
        line.set_ydata(chart_y)

    previous_step = step
    step = (step + 24) % frame.nobs


fname1 = "LSOA_2004_London_Low_Resolution.shp"
fname2 = "LSOA_2011_London_gen_MHW.shp"
fname3 = "London_Borough_Excluding_MHW.shp"  #
fname4 = "London_Ward.shp"
fname5 = "London_Ward_CityMerged.shp"
fname6 = "MSOA_2004_London_High_Resolution.shp"
fname7 = "MSOA_2011_London_gen_MHW.shp"
fname8 = "OA_2011_London_gen_MHW.shp"

fname = Path(f"statistical-gis-boundaries-london/ESRI/{fname2}")
wow = "Best_Signals_wow_2018_2019_timestamped_and_cleaned.nc"
density = "Best_Signals_wow_london_density.nc"

line = coastline_mesh_line(str(fname))
mesh = coastline_mesh(str(fname))
n_faces = mesh.n_faces
data = np.ones(n_faces) * np.nan
mesh.cell_data["ids"] = np.arange(n_faces)
mesh.cell_data["data"] = data

frame = prepare(wow, density, mesh)

emesh = mesh.extrude((0, 0, -EXTRUDE_BASE), capping=True)
tmesh = emesh.triangulate()

interval = 200
previous_step = 0
step = 12

plotter = BackgroundPlotter()
title = "Temperature / K"
sargs = dict(title=title, shadow=True, nan_annotation=True)
plotter.add_mesh(
    tmesh,
    scalar_bar_args=sargs,
    cmap="balance",
    clim=frame.clim,
    nan_color="grey",
    show_scalar_bar=False,
)
color = "darkgrey"
plotter.add_mesh(line, color=color)
plotter.add_mesh(line.translate((0, 0, -EXTRUDE_BASE), inplace=False), color=color)

for cid in frame.extruded:
    plotter.add_mesh(
        frame.extruded[cid],
        scalar_bar_args=sargs,
        cmap="balance",
        clim=frame.clim,
        nan_color="cyan",
        show_scalar_bar=True,
    )

actor = plotter.scalar_bar.GetAnnotationTextProperty()
actor.SetFontSize(5)

plotter.add_text(
    f"{clean(fname.stem)} ({1000 // interval}Hz refresh)",
    position="upper_left",
    font_size=10,
    shadow=True,
)
actor = plotter.add_point_labels(
    frame.pois,
    frame.sites,
    point_color="red",
    point_size=5,
    render_points_as_spheres=True,
    always_visible=True,
    shape="rounded_rect",
    shape_opacity=0.3,
    font_size=10,
)

dt_actor = plotter.add_text(
    frame.dt[step], position="upper_right", font_size=10, shadow=True
)

fig, ax = plt.subplots(tight_layout=True)
size = 7
ax.set_xlabel("Date", fontsize=size)
ax.set_ylabel("Mean Temperature / K", fontsize=size)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.yaxis.set_major_locator(MaxNLocator(7))
start = datetime.strptime(frame.dt[0], "%Y-%m-%d %H:%M:%S")
end = datetime.strptime(frame.dt[-1], "%Y-%m-%d %H:%M:%S")
ax.set_xlim([start, end])
ax.set_ylim([frame.data.min(), frame.data.max()])
ax.tick_params(axis="x", labelsize=size - 1, labelrotation=45)
ax.tick_params(axis="y", labelsize=size)
ymean = frame.data.mean()
color = "red"
ax.hlines(y=ymean, xmin=start, xmax=end, color=color, linewidth=1)
# https://stackoverflow.com/questions/6319155/show-the-final-y-axis-value-of-each-line-with-matplotlib
plt.annotate(
    f"{ymean:.1f}",
    xy=(1, ymean),
    xytext=(6, 0),
    xycoords=("axes fraction", "data"),
    textcoords="offset points",
    fontsize=size,
    color=color,
    va="center",
    bbox=dict(boxstyle="round", ec=color, fc="darkgrey", lw=1),
)
ax.grid(True)
chart = pv.ChartMPL(fig, size=(0.3, 0.35), loc=(0.01, 0.01))
chart.background_color = (1, 1, 1, 0.5)
plotter.add_chart(chart)
line, chart_x, chart_y = None, None, None

# plotter.add_axes()
plotter.view_xy()

plotter.add_callback(callback, interval=interval)
