# Icosphere

![icosphere](https://raw.githubusercontent.com/bjlittle/collab/main/informatics-lab/icosphere/assets/icosphere.png)

Proof of concept playground for icosphere rendering with `geovista`.

### Contents

Refer to the `main.py` script for various example techniques to render the icosphere using `geovista`.

`networkx` is used to detect graph cycles and derive the mesh connectivity from the adjacency matrix. I'm sure there
are other ways to do this, but this seems to (naively) work for now as a first approach.

The derived connectivity is then used to create a `pyvista` mesh using `geovista`.

### Install

Simply install `geovista` using either `conda` or `pip`, then run the `main.py` script to play.

For `geovista` installation instructions, please see [here](https://github.com/bjlittle/geovista#installation) 👍

### Play

Simply,
```bash
python main.py
```
Enjoy 🥳


### Reference

- [GeoVista](https://github.com/bjlittle/geovista)
- [PyVista](https://github.com/pyvista/pyvista) [[docs](https://docs.pyvista.org/)]
