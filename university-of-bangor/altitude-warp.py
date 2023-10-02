import geovista
import geovista.theme
import netCDF4 as nc
import numpy as np
import numpy.ma as ma


fname = "kent_estuary_merged_map.nc"
ds = nc.Dataset(fname)

node_x = ds.variables["mesh2d_node_x"][:]
node_y = ds.variables["mesh2d_node_y"][:]
node_z = ds.variables["mesh2d_node_z"]

connectivity = ds.variables["mesh2d_face_nodes"][:]
# fix invalid connectivity indices
connectivity = ma.masked_where(connectivity < 0, connectivity)
# this appears to be an invalid cell - should be a triangle not a quad?
connectivity[64129, 3] = ma.masked

mesh = geovista.Transform.from_unstructured(node_x, node_y, connectivity=connectivity)
mesh[scalars := "altitude"] = node_z[:]

mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
mesh.warp_by_scalar(scalars=scalars, inplace=True, factor=1e-5)

plotter = geovista.GeoPlotter()
sargs = {"title": f"{node_z.standard_name} / {node_z.units}", "shadow": True}
plotter.add_mesh(mesh, show_edges=True, smooth_shading=False, scalar_bar_args=sargs)
plotter.add_axes()
plotter.show()
