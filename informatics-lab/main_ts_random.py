from pathlib import Path
from typing import List, Optional

import cartopy.io.shapereader as shp
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

import geovista.theme


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
                coords = geometry.exterior.coords if isinstance(geometry, Polygon) else geometry.coords
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
    return mesh
    

def callback() -> None:
    global tmesh
    global n_faces

    indices = tmesh.cell_data["ids"]
    data = np.random.randint(0, n_faces, n_faces)
    tmesh.cell_data["data"] = data[indices]
    tmesh.set_active_scalars("data", preference="cell")


fname1 = "LSOA_2004_London_Low_Resolution.shp"
fname2 = "LSOA_2011_London_gen_MHW.shp"
fname3 = "London_Borough_Excluding_MHW.shp"
fname4 = "London_Ward.shp"
fname5 = "London_Ward_CityMerged.shp"
fname6 = "MSOA_2004_London_High_Resolution.shp"
fname7 = "MSOA_2011_London_gen_MHW.shp"
fname8 = "OA_2011_London_gen_MHW.shp"

fname = Path(f"statistical-gis-boundaries-london/ESRI/{fname2}")

line = coastline_mesh_line(str(fname))
mesh = coastline_mesh(str(fname))
n_faces = mesh.n_faces
data = np.random.randint(0, n_faces, n_faces)
mesh.cell_data["data"] = data
mesh.cell_data["ids"] = np.arange(n_faces)
tmesh = mesh.triangulate()

interval = 200

plotter = BackgroundPlotter()
sargs = dict(title="Random Data", shadow=True)
plotter.add_mesh(tmesh, scalar_bar_args=sargs)
#plotter.add_mesh(line) #, color="grey")
plotter.add_text(
    f"{fname.name} ({1000 // interval}Hz refresh)",
    position="upper_left",
    font_size=10,
    shadow=True,
)
plotter.add_axes()
plotter.view_xy()

plotter.add_callback(callback, interval=interval)
