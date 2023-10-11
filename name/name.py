from geovista.qt import GeoBackgroundPlotter
import geovista.theme
import iris
import cmocean


def callback() -> None:
    global tstep
    global plotter
    global frames_by_time
    global levels_by_time
    global actors
    global n_steps
    global cmap
    global zfactor
    global time
    global tactor
    global fmt
    global sargs

    previous = set(levels_by_time[tstep])
    tstep = (tstep + 1) % n_steps
    current = set(levels_by_time[tstep])
    remove = previous - current
    render = {}
    for level, frame in zip(levels_by_time[tstep], frames_by_time[tstep]):
        actor = plotter.add_mesh(
            frame,
            cmap=cmap,
            clim=clim,
            zlevel=(level + 1) * zfactor,
            name=f"mesh{level}",
            show_scalar_bar=True,
            scalar_bar_args=sargs,
        )
        render[level] = actor
    for level in remove:
        plotter.remove_actor(actors[level], render=True)
    actors = render
    tactor.SetText(3, time.units.num2date(time[tstep].points)[0].strftime(fmt))


fname = "./assets/QVA_grid1_4D.nc"

cube = iris.load_cube(fname)
lons = cube.coord("longitude")
lats = cube.coord("latitude")
time = cube.coord("time")

if not lons.has_bounds():
    lons.guess_bounds()

if not lats.has_bounds():
    lats.guess_bounds()

clons = lons.contiguous_bounds()
clats = lats.contiguous_bounds()

mesh = geovista.Transform.from_1d(clons, clats)

levels_by_time = {}
frames_by_time = {}
actors = {}
n_steps = cube.shape[0]
cmin = 0.00001

for tstep in range(n_steps):
    frames = frames_by_time.setdefault(tstep, [])
    levels = levels_by_time.setdefault(tstep, [])
    print("\n")
    for level in range(cube.shape[1]):
        mesh["data"] = cube[tstep, level].data.ravel()
        tmesh = mesh.threshold((cmin, 1))
        if tmesh.n_cells:
            print(f"{tstep=}, {level=}, {tmesh.n_cells}")
            frames.append(tmesh)
            levels.append(level)

cmap = cmocean.tools.crop_by_percent(cmap=cmocean.cm.gray_r, per=10, which="max")
clim = (cmin, 0.26757073)
clim = (cmin, 0.02)
zfactor = 10

fmt = "%Y-%m-%d %H:%M"

name = cube.name().replace("_", " ")
name = " ".join([word.capitalize() for word in name.split(" ")])
units = str(cube.units).replace(" ", "")
title = f"{name} / {units}"
sargs = {"title": title}

plotter = GeoBackgroundPlotter()
plotter.add_coastlines()
plotter.add_base_layer(texture=geovista.natural_earth_1(), zlevel=0)
plotter.add_axes()

tstep = 0
for level, frame in zip(levels_by_time[tstep], frames_by_time[tstep]):
    actor = plotter.add_mesh(
        frame,
        cmap=cmap,
        clim=clim,
        zlevel=(level + 1) * zfactor,
        name=f"mesh{level}",
        show_scalar_bar=True,
        scalar_bar_args=sargs,
    )
    actors[level] = actor

text = time.units.num2date(time[tstep].points)[0].strftime(fmt)
tactor = plotter.add_text(text, position="upper_right", font_size=10)

plotter.add_callback(callback, interval=250)
