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
