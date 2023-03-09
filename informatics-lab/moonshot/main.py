import warnings

import geovista
import netCDF4 as nc
import networkx as nx
import numpy as np
from geovista import theme  # noqa: F401
from numpy.typing import ArrayLike


def build_connectivity(matrix: ArrayLike) -> ArrayLike:
    assert matrix.ndim == 2
    M, N = matrix.shape
    assert M == N

    midx, nidx = np.where(matrix > 0)
    idxs = set(midx) | set(nidx)
    edges = list(zip(midx, nidx))
    graph = nx.Graph()
    [graph.add_edge(*edge) for edge in edges]

    faces = set()
    n_verts = 3
    for idx in sorted(idxs):
        cycles = nx.cycle_basis(graph, idx)
        cycles = {
            tuple(sorted(cycle))
            for cycle in filter(lambda cycle: len(cycle) == n_verts, cycles)
        }
        faces.update(cycles)

    if missing := set(range(N)) - idxs:
        wmsg = f"following matrix node indices are missing: {missing}"
        warnings.warn(wmsg)

    print(f"matrix derived mesh contains {len(faces)} faces")
    connectivity = np.array([face for face in faces])

    return connectivity


# load the variables from the netcdf file
fname = "graph_adjacency_matrix.nc"
ds = nc.Dataset(fname)
lat = ds.variables["lat"][:]
lon = ds.variables["lon"][:]
matrix = ds.variables["__xarray_dataarray_variable__"][:]

# create the mesh connectivity from the adjacency matrix
connectivity = build_connectivity(matrix)

# create the mesh
mesh = geovista.Transform.from_unstructured(lon, lat, connectivity=connectivity)

mesh["data"] = np.random.random(mesh.n_points)

# render the mesh
plotter = geovista.GeoPlotter(crs="+proj=eqc")
plotter.add_mesh(mesh, style="wireframe", line_width=3, color="black")
plotter.add_base_layer(
    mesh=mesh, texture=geovista.natural_earth_hypsometric(), zlevel=-10
)
plotter.add_points(mesh, color="red", render_points_as_spheres=True)
plotter.add_axes()
plotter.show()
