import geovista
import geovista.theme

from geovista.geodesic import BBox
from geovista.geometry import coastlines
from geovista.pantry.meshes import regular_grid
from geovista.core import add_texture_coords
from geovista.raster import wrap_texture


bbox = BBox([-1.0, 0.9, 0.9, -1.0], [52.9, 52.9, 51.5, 51.5])
grid = regular_grid("r2000")

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