First, create and activate the `conda` environment for the `jupyter notebook` as follows:
```bash
> conda env create --file environment.yml
> conda activate ucar-dev
```

Now, decompress the data:
```bash
> bzip2 -d data
```

Finally, start the `jupyter notebook` server:
```bash
> jupyter notebook
```

Alternatively, you can start the `jupyter notebook` in the background and connect to it remotely. 

On executing `jupyter`, it will inform you of the URL and port to connect. This is useful is you wish
to stand-up the server but connect to it from a host/client with a GPU, if the server host doesn't
have one:
```bash
> jupyter notebook --no-browse --ip=0.0.0.0
```

Note that, if you wish to add extra `conda` packages to the `ucar-dev` environment, then simply
do the following when the `ucar-dev` environment is activated:
```shell
> conda install --channel conda-forge <package> ...
```
Alternatively, simply recreate the **entire environment again** but with the additional packages
included in the `environment.yml`. Typically, this is better for the `conda` package resolver,
otherwise it's easy to eventually end up with a broken environment by installing additional
packages into an already existing environment in a piecemeal fashion.
