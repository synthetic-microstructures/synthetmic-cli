import click
import pandas as pd
import pyfiglet

from synthetmic_cli import commands
from synthetmic_cli.shared import (
    PropertyExtension,
    color_text,
    describe_df,
    validate_path,
)


@click.group(
    help=f"""\b
\033[36m{pyfiglet.figlet_format("SynthetMic-CLI")}\033[0m 
SynthetMic-CLI is a command line interface for generating 2D and 3D
synthetic polycrystalline microstructures using Laguerre tessellations.
It uses the fast algorithms developed in this paper:

Bourne, D.P., Kok, P.J.J., Roper, S.M., and Spanjer, W.D.T. (2020),
‘Laguerre Tessellations and Polycrystalline Microstructures: a Fast
Algorithm for Generating Grains of Given Volumes’, Philosophical Magazine,
100, 2677–2707

for generating grains of prescribed volumes using optimal transport theory.
It is built on top of SynthetMic package which is the Python implementation of
the fast algorithms.

Please report any feedback or bugs to the maintainers here:
    https://github.com/synthetic-microstructures/synthetmic-cli/issues

The repository for this object can be found at:
    https://github.com/synthetic-microstructures/synthetmic-cli

Authors: Rasheed Ibraheem, David Bourne, and Steven Roper.
"""
)
def cli() -> None:
    pass


cli.add_command(commands.sample_seeds)
cli.add_command(commands.sample_spvols)
cli.add_command(commands.sample_dpvols)
cli.add_command(commands.generate)


@cli.command(
    help="""
Print out information about sampled seeds or target volumes saved in PATH.

Note that the sampled data file must be in either csv or txt format.
    """
)
@click.argument(
    "path",
    type=click.Path(exists=True),
    nargs=1,
)
def describe(path: str) -> None:
    validate_path(path, allowed_file_ext=[ext for ext in PropertyExtension])

    click.echo(
        color_text(
            f"* Describing {path}...",
            color="green",
            bold=True,
        ),
        color=True,
    )

    df = pd.read_csv(path, index_col=False)
    describe_df(df)

    return None
