from pathlib import Path

from cf_units import Unit
import cmocean
import geovista
import geovista.theme
import iris
import netCDF4 as nc
import numpy as np
import pyvista as pv


def make_points(n=100):
    """Helper to make XYZ points"""
    theta = np.linspace(-1 * np.pi, 1 * np.pi, n)
    z = np.linspace(2, -1, n)
    zz = np.concatenate([np.linspace(-3, 0, n//3), np.zeros(2*(n//3) + (n%3))])
    r = zz**2 + 4
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    return np.column_stack((x, y, z))


spline = pv.Spline(make_points(), 960) 

mesh = geovista.samples.lfric_sst()

plotter = geovista.GeoPlotter(off_screen=True)

plotter.open_movie("spline.mp4")

plotter.add_mesh(
    mesh, show_scalar_bar=False, lighting=False
)
plotter.set_position(spline.points[0])
plotter.set_focus((0, 0, 0))
plotter.set_viewup((0, 0, 1))
plotter.add_coastlines(color="black")
plotter.set_background(color="black")
plotter.add_base_layer(texture=geovista.blue_marble())
plotter.add_axes(color="white")

plotter.show(auto_close=False)

plotter.write_frame()

for i, point in enumerate(spline.points[1:]):
    print(i)

    plotter.set_position(point)
    plotter.write_frame()

plotter.close()
