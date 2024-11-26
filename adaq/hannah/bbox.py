import geovista
import geovista.theme

from geovista.geodesic import BBox
from geovista.geometry import coastlines
from geovista.pantry.meshes import lfric


bbox = BBox([-13.4, 3.4, 3.4, -13.4], [59.2, 59.2, 49.75, 49.75])

coasts = bbox.enclosed(coastlines(zlevel=0))
base = bbox.enclosed(lfric(), preference="point")

p = geovista.GeoPlotter()
p.add_mesh(coasts, color="black")
p.add_mesh(base, opacity=0.5, show_edges=True)
p.add_mesh(bbox.boundary(), color="green", line_width=3)
p.add_axes()
p.show()