#!/usr/bin/env python3
# Copyright (c) 2021, GeoVista Contributors.
#
# This file is part of GeoVista and is distributed under the 3-Clause BSD license.
# See the LICENSE file in the package root directory for licensing details.

"""
OISST AVHRR Grid
----------------

This example demonstrates how to render a rectilinear grid.

📋 Summary
^^^^^^^^^^

Creates a mesh from 1-D latitude and longitude rectilinear cell bounds.

The resulting mesh contains quad cells.

The example uses NOAA/NECI 1/4° Daily Optimum Interpolation Sea Surface Temperature
(OISST) v2.1 Advanced Very High Resolution Radiometer (AVHRR) gridded data.
The data targets the mesh faces/cells.

Note that, a threshold is also applied to remove land ``NaN`` cells, and a
NASA Blue Marble base layer is rendered along with Natural Earth coastlines.

.. tags::

    component: coastlines, component: texture,
    domain: oceanography,
    filter: threshold,
    load: rectilinear

----

"""  # noqa: D205,D212,D400

from __future__ import annotations

import geovista as gv
from geovista.pantry.data import oisst_avhrr_sst
import geovista.theme
import numpy as np
import pyvista as pv


def make_points(n=100):
    """Helper to make XYZ points"""
    theta = np.linspace(-1 * np.pi, 1 * np.pi, n)
    theta = np.concatenate([theta, theta[1:]])
    z = np.linspace(4, -2, n*2-1)
    #z = np.concatenate([z, z[::-1][1:]])
    zz = np.concatenate([np.linspace(-3, 0, n//3), np.zeros(2*(n//3) + (n%3))])
    zz = np.concatenate([zz, zz[::-1][1:]])
    r = zz**2 + 5
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    return np.column_stack((x, y, z))


def main() -> None:
    """Plot an OISST AVHRR rectilinear grid.

    Notes
    -----
    .. versionadded:: 0.1.0

    """
    # Load the sample data.
    sample = oisst_avhrr_sst()

    # Create the mesh from the sample data.
    mesh = gv.Transform.from_1d(sample.lons, sample.lats, data=sample.data)

    # Remove cells from the mesh with NaN values.
    mesh = mesh.threshold()

    stripes = "GLOBE---1850-2022-MO.png"
    tex = pv.read_texture(stripes)

    # Plot the rectilinear grid.
    p = gv.GeoPlotter(off_screen=True)

    n_steps = 500
    spline = pv.Spline(make_points(), n_steps)

    p.open_movie("globe-orbit.mp4")
    p.set_position(spline.points[0])
    p.set_focus((0, 0, 0))
    p.set_viewup((0, 0, 1))

    sargs = {"title": f"{sample.name} / {sample.units}", "shadow": True}
    mesh.set_active_scalars(None)
    p.add_mesh(mesh, texture=tex, smooth_shading=True)
    p.add_base_layer(texture=gv.black_marble(), smooth_shading=True)
    p.add_coastlines(color="lightgray")
    p.add_axes(color="white")
    p.add_text(
        "#ShowYourStripes",
        position="upper_left",
        font_size=15,
        shadow=True,
        color="lightgray"
    )
    p.add_background_image(stripes)

    p.show(auto_close=False)

    for step in range(n_steps):
        print(f"frame = {step + 1} of {n_steps}")
        p.set_position(spline.points[step])
        p.write_frame()

    p.close()



if __name__ == "__main__":
    main()
