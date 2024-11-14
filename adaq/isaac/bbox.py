import geovista
import geovista.theme

from geovista.geodesic import BBox
from geovista.geometry import coastlines
from geovista.pantry.meshes import regular_grid
from geovista.core import add_texture_coords
from geovista.raster import wrap_texture


bbox = BBox([124.0, 148.0, 148.0, 124.0], [47.0, 47.0, 30.0, 30.0])
grid = regular_grid("r200")

coasts = bbox.enclosed(coastlines(zlevel=0))
base = bbox.enclosed(grid, preference="point")

base = add_texture_coords(base)
texture = wrap_texture(geovista.natural_earth_hypsometric())

p = geovista.GeoPlotter()
p.set_background(color="black")
p.add_mesh(coasts, color="lightgray")
p.add_mesh(base, texture=texture, opacity=0.5, show_edges=True)
p.view_poi()
p.add_mesh(bbox.boundary(), color="orange", line_width=3)
p.add_axes()
p.show()