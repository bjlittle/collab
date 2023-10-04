import cmocean
import geovista
import iris


fname = "/project/kscale/DATA/outdir_20200120T0000Z/u-co997_RAL3_n1280_DW/single_olwr/20200120_20200120T0000Z_global_olwr_hourly.nc"

cube = iris.load_cube(fname)
lons = cube.coord("longitude")
lats = cube.coord("latitude")

mesh = geovista.Transform.from_1d(lons.bounds, lats.bounds, data=cube[0].data, name="data")

mesh = mesh.threshold(240, method="lower")

plotter = geovista.GeoPlotter(crs="+proj=robin")
cmap = "Blues"
cmap = cmocean.cm.gray_r
plotter.add_mesh(mesh, cmap=cmap, show_scalar_bar=False, lighting=False)
plotter.add_coastlines(color="black")
#plotter.add_graticule(lon_step=5, lat_step=5, show_labels=False, mesh_args={"color": [0.4, 0.4, 0.4]})
plotter.set_background(color="black")
plotter.add_base_layer(texture=geovista.natural_earth_1())
plotter.view_xy()
plotter.camera.zoom(1.5)
plotter.show()
