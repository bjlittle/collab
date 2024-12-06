import iris
import netCDF4 as nc
import numpy as np
import pyvista as pv

from geovista.common import to_cartesian

factor = 4e-6

p_fname = "data/Fields_grid3_C1.nc"
p_cube = iris.load_cube(p_fname)

p_t = p_cube.coord("time")
p_z = p_cube.coord("height")
p_y = p_cube.coord("latitude")
p_x = p_cube.coord("longitude")

p_ds = nc.Dataset(p_fname)
p_data = p_ds.variables["PM10_AIR_CONCENTRATION"]

p_y_cb = p_y.contiguous_bounds()
p_x_cb = p_x.contiguous_bounds()
p_z_cb = p_z.contiguous_bounds()

p_xx, p_yy, p_zz = np.meshgrid(p_x_cb, p_y_cb, p_z_cb, indexing="ij")
shape = p_xx.shape

p_xyz = to_cartesian(p_xx, p_yy, zlevel=p_zz, zscale=factor)

plume = pv.StructuredGrid(p_xyz[:, 0].reshape(shape), p_xyz[:, 1].reshape(shape), p_xyz[:, 2].reshape(shape))

plume.save("plume.vtk")