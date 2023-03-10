import warnings
from typing import Optional

import geovista
import netCDF4 as nc
import networkx as nx
import numpy as np
from geovista import theme  # noqa: F401
from numpy.typing import ArrayLike
from pyvista import PolyData

#: Default to an icosphere mesh composed of triangular faces i.e., 3 vertices
DEFAULT_N_VERTS: int = 3


def build_connectivity(matrix: ArrayLike, n_verts: Optional[int] = None) -> ArrayLike:
    """
    Create the mesh connectivity based on the adjacency `matrix`, which
    encodes the mesh topology as the edges connecting joining immediate
    graph node neighbours.

    Parameters
    ----------
    matrix : ndarray of int
        A 2-D ``(N, N)`` square bit-mask array, representing the edge
        connectivity between the ``N`` nodes (or vertices) in the graph
        (or mesh). Where each row represents a specific node and its
        direct connectivity to any of the other ``N - 1`` nodes in the
        graph.

        e.g., The following simple example matrix represents a graph
        containing 3 nodes connected by 3 edges to form a resultant
        mesh with a single triangular face.

           0 1 2      0
                      |\
        0  0 1 1      | \
        1  1 0 1  ->  |  \
        2  1 1 0      2---1

        In terms of edge-to-node pairs we have the following:

        (0, 1), (0, 2), (1, 0), (1, 2), (2, 0) and (2, 1)
    n_verts : int, optional
        The number of vertices (or nodes) that define the geometry
        of a mesh face. Default to ``DEFAULT_N_VERTS``.

    Returns
    -------
    ndarray
        A 2-D array of vertex offsets per face e.g., an icosphere
        composed of ``N`` faces will result in an array of shape
        ``(N, 3)``.

    """
    # sanity check the matrix i.e., 2-D and square
    assert matrix.ndim == 2
    M, N = matrix.shape
    assert M == N

    if n_verts is None:
        n_verts = DEFAULT_N_VERTS

    # for simplicity convert to boolean mask
    matrix = matrix.astype(bool)

    # construct a networkx graph from edge-to-node connectivity
    # derived from the adjacency matrix
    midx, nidx = np.where(matrix)
    idxs = set(midx) | set(nidx)
    edges = list(zip(midx, nidx))
    graph = nx.Graph()
    [graph.add_edge(*edge) for edge in edges]

    # use networkx to discover all cycles within the graph composed
    # of n_verts nodes.
    faces = set()
    for idx in sorted(idxs):
        cycles = nx.cycle_basis(graph, idx)
        cycles = {
            tuple(sorted(cycle))
            for cycle in filter(lambda cycle: len(cycle) == n_verts, cycles)
        }
        faces.update(cycles)

    # sanity check all graph node indices participate in a mesh face or
    # graph node cycle.
    if missing := set(range(N)) - idxs:
        wmsg = f"the following matrix node indices are missing: {missing}"
        warnings.warn(wmsg)

    connectivity = np.array([face for face in faces])

    return connectivity


def plot_point_cloud(mesh: PolyData) -> None:
    # copy the mesh to decorate it with data,
    # but with no side-effects to the caller mesh
    mesh = mesh.copy()
    mesh["Sample Data"] = np.random.random(mesh.n_points)

    plotter = geovista.GeoPlotter()
    plotter.add_points(mesh, render_points_as_spheres=True)
    plotter.add_axes()
    plotter.add_text(
        "Graph Point Cloud",
        position="upper_left",
        font_size=10,
        shadow=True,
    )
    plotter.show()


def plot_surface(mesh: PolyData, preference: Optional[str] = None) -> None:
    extra = ""
    mesh = mesh.copy()

    if preference is not None:
        preference = str(preference).lower()

    if preference == "face":
        mesh["Sample Data (Face)"] = np.random.random(mesh.n_cells)
        extra = "with Data on the Mesh Faces"

    if preference == "point":
        mesh["Sample Data (Point)"] = np.random.random(mesh.n_points)
        extra = "with Data on the Mesh Points"

    plotter = geovista.GeoPlotter()
    plotter.add_mesh(mesh, show_edges=True)
    plotter.add_axes()
    plotter.add_text(
        f"Surface {extra}",
        position="upper_left",
        font_size=10,
        shadow=True,
    )
    plotter.show()


def plot_wireframe(mesh: PolyData) -> None:
    plotter = geovista.GeoPlotter()
    plotter.add_mesh(mesh, style="wireframe", color="black")
    plotter.add_axes()
    plotter.add_text(
        "Wireframe",
        position="upper_left",
        font_size=10,
        shadow=True,
    )
    plotter.show()


def plot_labels(mesh: PolyData, preference: str) -> None:
    assert preference in ["face", "point"]

    plotter = geovista.GeoPlotter()
    plotter.add_mesh(mesh, show_edges=True)

    if preference == "point":
        labels = [f"{i}" for i in range(mesh.n_points)]
        plotter.add_point_labels(mesh.points, labels)
    else:
        labels = [f"{i}" for i in range(mesh.n_cells)]
        centers = mesh.cell_centers()
        plotter.add_point_labels(centers.points, labels)

    plotter.add_axes()
    plotter.add_text(
        f"Surface with {preference.capitalize()} Labels",
        position="upper_left",
        font_size=10,
        shadow=True,
    )
    plotter.show()


def plot_base_layer(mesh: PolyData, zlevel: Optional[int] = None) -> None:
    if zlevel is None:
        zlevel = -1
    else:
        # enforce a negative level here
        zlevel = -abs(int(zlevel))

    plotter = geovista.GeoPlotter()
    plotter.add_mesh(mesh, style="wireframe", color="black")
    plotter.add_points(mesh.points, render_points_as_spheres=True, color="red")
    plotter.add_base_layer(
        mesh=mesh.copy(), texture=geovista.natural_earth_hypsometric(), zlevel=zlevel
    )
    plotter.add_axes()
    plotter.add_text(
        f"Wireframe with Points, and Texture Mapped Base Layer ({zlevel=})",
        position="upper_left",
        font_size=10,
        shadow=True,
    )
    plotter.show()


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

#
# some example plots using geovista...
#

plot_point_cloud(mesh)

for preference in [None, "face", "point"]:
    plot_surface(mesh, preference=preference)

plot_wireframe(mesh)

for preference in ["point", "face"]:
    plot_labels(mesh, preference)

for zlevel in [-1, -10, -100, -300]:
    plot_base_layer(mesh, zlevel=zlevel)
