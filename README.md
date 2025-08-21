# SynthetMic-CLI

### About
SynthetMic-CLI is a command line interface for generating 2D and 3D synthetic polycrystalline microstructures using Laguerre tessellations.
It uses the fast algorithms (developed in this [paper](https://www.tandfonline.com/doi/full/10.1080/14786435.2020.1790053))
for generating grains of prescribed volumes using optimal transport theory. It is built on
top of [SynthetMic](https://github.com/synthetic-microstructures/synthetmic) package which is the Python implementation of the fast algorithms.

### Installation
To install the latest version of the CLI via `pip`, run
```
pip install synthetmic-cli
```
> If you are using `uv` to manage your project, run the following command instead:
>
> ```uv add synthetmic-cli```

### Usage
All the available commands in the CLI can be checked by running
```
sm --help
```
Each of these commands are explained as follows.

1. `sample-seeds` samples random seeds in a box with dimension specfied by `BOX_SPECS`. For instance, to sample 1000 random seeds in a unit cube (this means seeds will be generated in the domain [0, 1] x [0, 1] x [0, 1]) and save the results in the path ./mydir/seeds.csv, run
```
sm sample-seeds 1 1 1 --n-grains 1000 --save-path ./mydir/seeds.csv
```
> You can check all other options under this command by running
>
> ```sm sample-seeds --help```

1. `sample-spvols` samples single phase target volumes using box dimension given by `BOX_SPECS`. By default, this command generates constant target volumes, but users can choose from other distributions including uniform and log-normal distributions.

For example, to generate constant target volumes for 1000 grains in a unit cube and save the results to ./mydir/vols.csv, one runs
```
sm sample-spvols 1 1 1 --n-grains 1000 --save-path ./mydir/spvols.csv
```
> Other options under this command can be checked by running
>
> ```sm sample-vols --help```

1. `sample-dpvols` samples dual phase targe volumes in a box specified by `BOX_SPECS`. This command allows specifying different volume distribution for each phase of the microstructure. For instance, one can specify uniform distribution on phase 1 and log-normal for phase 2 in a unit cube of 1000 grains:
```
sm sample-dpvols 1 1 1 --n-grains 1000 --dist-params1 uniform 1 2 --dist-params2 lognormal 1 0.35 --save-path ./mydir/dpvols.csv
```
> Note: `--dist-params1 uniform 1 2` means sample from uniform distribution with parameters a = 1 and b = 2 for phase 1; `--dist-params2 lognormal 1 0.35` means sample from lognormal distribution with parameters mean = 1 and std = 0.35 for phase 2.
>
> Other options under this command can be checked by running
> 
> ```sm sample-dpvols --help```

1. `describe` prints out information about seeds or target volumes saved in `PATH`.

For instance, to print out information about seeds saved in ./mydir/seeds.csv, run
```
sm describe ./mydir/seeds.csv
```

1. `generate` generates synthetic microstructure with box specs `BOX_SPECS`, seeds saved in `SEEDS_PATH`, and target volumes saved in `TARGET_VOLS_PATH`.

For example, to generate microstructure in a unit cube with seeds in ./mydir/seeds.csv and target volumes in ./mydir/dpvols.csv; and save all results to ./mydir/results.zip, we run
```
sm generate 1 1 1 ./mydir/seeds.csv ./mydir/dpvols.csv --save-path ./mydir/results.zip
```
> To see all options under this command, run 
>
> ```sm generate --help```

### Build from source
If you would like to build this project from source either for development purposes or for any other reason, it is recommended to install [uv](https://docs.astral.sh/uv/). This is what is adopted in this project. To install uv, follow the instructions in this [link](https://docs.astral.sh/uv/getting-started/installation/).

If you don't want to use uv, you can use other alternatives like [pip](https://pip.pypa.io/en/stable/).

The following instructions use uv for set up.

1. Clone the repository by running

    ```
    git clone https://github.com/synthetic-microstructures/synthetmic-cli
    ```

1. Create a python virtual environment by running

    ```
     uv venv .venv --python PYTHON_VERSION
    ```
    > Here, PYTHON_VERSION is the supported Python version. Note that this project requires version >=3.11

1. Activate the virtual environment by running

    ```
    source .venv/bin/activate
    ```

1. Prepare all modules and dependencies by running the following:

    ```
    uv sync --all-extras
    ```
    ```

### Authors and maintainers
- [R. O. Ibraheem](https://github.com/Rasheed19)
- [D. P. Bourne](https://github.com/DPBourne)
- [S. M. Roper](https://github.com/smr29git)

