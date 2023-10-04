import iris
import numpy as np
import numpy.ma as ma
import pyvista as pv

import geovista
import geovista.theme


def tidy(name):
    name = name.replace("_", " ")
    parts = name.split()
    return " ".join([part.capitalize() for part in parts])


names = [
    "potential temperature",
    "sound_speed_in_sea_water",
    "salinity",
]

colors = [
    "red",
    "green",
    "blue",
]

class SetVisibilityCallback:
    def __init__(self, actor):
        self.actor = actor
    def __call__(self, state):
        self.actor.SetVisibility(state)


def load(name: str) -> pv.PolyData:
    fname = "./canny_hor.nc"
    cube = iris.load_cube(fname, name)

    lons = cube.coord("longitude").points
    lats = cube.coord("latitude").points

    z, m, n = ma.where(cube.data != 0)

    (idx,) = np.where(z < 50)
    m_idx = m[idx]
    n_idx = n[idx]
    z_idx = z[idx]

    lons_idx = lons[(m_idx, n_idx)]
    lats_idx = lats[(m_idx, n_idx)]

    #data = cube.data[(z_idx, m_idx, n_idx)]

    xyz = geovista.common.to_cartesian(lons_idx, lats_idx, zlevel=z_idx, zscale=-1e-3)

    cloud = pv.PolyData(xyz)
    cloud[name] = cloud.points[:, 2]
    #cloud["data"] = data

    print(name)
    print(cloud)

    return cloud


clouds = [load(name) for name in names]

p = geovista.GeoPlotter()

start = 12
size = 25

for color, name, cloud in zip(colors, names, clouds):
    actor = p.add_mesh(cloud, color=color, point_size=5)
    callback = SetVisibilityCallback(actor)
    p.add_checkbox_button_widget(
        callback,
        value=True,
        position=(5.0, start),
        size=size,
        border_size=2,
        color_on=color,
        color_off="grey",
        background_color="grey",
    )
    p.add_text(
        tidy(name),
        position=(35, start+5),
        color=color,
        font_size=8,
        shadow=True,
    )
    start += size + (size // 10)

actor = p.add_base_layer(texture=geovista.natural_earth_hypsometric(), opacity=0.5, zlevel=0)
callback = SetVisibilityCallback(actor)
p.add_checkbox_button_widget(
    callback,
    value=True,
    position=(5.0, start),
    size=size,
    border_size=2,
    color_on="orange",
    color_off="grey",
    background_color="grey",
)
p.add_text(
    "Base Layer",
    position=(35, start + 5),
    color="orange",
    font_size=8,
    shadow=True,
)

p.add_coastlines(color="black")
p.add_axes(interactive=True)
p.show()
