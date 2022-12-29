from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cartopy.io.shapereader as shp
import netCDF4 as nc
import numpy as np
import pyvista as pv
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
    singles: np.ndarray  # mask of single sites within a parent cell
    neighbours: dict  # the sites sharing which parent cell
    dt: np.ndarray  # time-series datetime strings
    data: np.ndarray  # contiguous time-series payload for each site
    clim: LIMIT  # data payload minimum and maximum


def clean(names):
    single = False
    if isinstance(names, str):
        names = [names]
        single = True
    cleaned = []
    for parts in names:
        parts = parts.replace("_", " ")
        cleaned.append(" ".join([part.capitalize() for part in parts.split()]))
    if single:
        result = cleaned[0]
    else:
        result = np.array(cleaned)
    return result


def prepare(fname: PathLike, mesh: pv.PolyData) -> Frame:
    ds = nc.Dataset(fname)

    sitename = ds.variables["SiteName"][:]
    data = ds.variables["Temperature_K"][:]
    datetime = ds.variables["DateTime"][:]
    lat = ds.variables["Lat"][:]
    lon = ds.variables["Lon"][:]

    sites, index = np.unique(sitename, return_index=True)
    sites = clean(sites)
    dt = np.unique(datetime)
    nobs = dt.size
    ilat, ilon = lat[index], lon[index]

    crs = CRS.from_wkt(str(mesh.field_data["crs"][0]))
    transformer = Transformer.from_crs(WGS84, crs, always_xy=True)
    xs, ys = transformer.transform(ilon, ilat, errcheck=False)
    zs = np.zeros_like(xs)
    shape = (-1, 1)
    pois = np.hstack([xs.reshape(shape), ys.reshape(shape), zs.reshape(shape)])

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
    singles = np.ones_like(cids, dtype=bool)
    if neighbours:
        singles[np.concatenate([*neighbours.values()])] = False

    clim = np.floor(data.min()), np.ceil(data.max())

    frame = Frame(
        nobs=nobs,
        index=index,
        cids=cids,
        sites=sites,
        pois=pois,
        singles=singles,
        neighbours=neighbours,
        dt=dt,
        data=data,
        clim=clim,
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


def callback() -> None:
    global step
    global data
    global frame
    global tmesh
    global dt_actor
    global av_actor
    global average
    global plotter

    obs = frame.data[frame.index + step]
    step_average = obs.mean()
    for cid, sites in frame.neighbours.items():
        cell_mean = np.mean(obs[sites])
        obs[sites] = cell_mean
    data[frame.cids] = obs

    indices = tmesh.cell_data["ids"]
    tmesh.cell_data["data"] = data[indices]
    tmesh.set_active_scalars("data", preference="cell")
    dt_actor.SetText(3, frame.dt[step])
    trend = ""
    if step_average > average:
        trend = "+"
    elif step_average < average:
        trend = "-"
    average = step_average
    av_actor.SetText(0, f"Mean={step_average:.2f}K ({trend})")

    step = (step + 24) % frame.nobs


fname1 = "LSOA_2004_London_Low_Resolution.shp"
fname2 = "LSOA_2011_London_gen_MHW.shp"
fname3 = "London_Borough_Excluding_MHW.shp"
fname4 = "London_Ward.shp"
fname5 = "London_Ward_CityMerged.shp"
fname6 = "MSOA_2004_London_High_Resolution.shp"
fname7 = "MSOA_2011_London_gen_MHW.shp"
fname8 = "OA_2011_London_gen_MHW.shp"

fname = Path(f"statistical-gis-boundaries-london/ESRI/{fname3}")
wow = "Best_Signals_wow_2018_2019_timestamped_and_cleaned.nc"

line = coastline_mesh_line(str(fname))
mesh = coastline_mesh(str(fname))
n_faces = mesh.n_faces
data = np.ones(n_faces) * np.nan
mesh.cell_data["data"] = data
mesh.cell_data["ids"] = np.arange(n_faces)

frame = prepare(wow, mesh)

tmesh = mesh.triangulate()

interval = 200
step = 12
average = 0

plotter = BackgroundPlotter()
sargs = dict(title="Temperature / K", shadow=True)
plotter.add_mesh(
    tmesh, scalar_bar_args=sargs, cmap="balance", clim=frame.clim, nan_color="grey"
)
plotter.add_mesh(line, color="black")
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
    point_size=10,
    render_points_as_spheres=True,
    always_visible=True,
    shape="rounded_rect",
    shape_opacity=0.3,
    font_size=10,
)

dt_actor = plotter.add_text(
    frame.dt[step], position="upper_right", font_size=10, shadow=True
)
av_actor = plotter.add_text("", position="lower_left", font_size=10, shadow=True)

plotter.add_axes()
plotter.view_xy()

plotter.add_callback(callback, interval=interval)
