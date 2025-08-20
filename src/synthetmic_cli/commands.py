import pathlib
from datetime import datetime
from typing import Any

import click
import numpy as np
import pandas as pd

import synthetmic_cli.shared as sd


def generate_dist_kwargs(*args: tuple[str, float, float]) -> dict[str, float]:
    dist = args[0]
    match dist:
        case sd.Distribution.LOGNORMAL:
            return dict(zip(["mean", "std"], args[1:]))

        case sd.Distribution.UNIFORM:
            return dict(zip(["low", "high"], args[1:]))

        case _:
            raise ValueError(
                f"Invalid dist: {dist}; value must be one of [{', '.join(sd.Distribution)}]."
            )


def get_dist_params_specs(prefix: str = "") -> dict:
    return dict(
        nargs=3,
        type=click.Tuple([str, float, float]),
        help=f"""{prefix}
        The volume distribution and its parameters. This must be entered with the following format:

            dist param1 param2

        where

        dist is the distribution name and must one of [{", ".join(sd.Distribution)}];\n
        param1, param2 are the parameters of the supported distribution.

        For instance,

            uniform 1.0 2.0

        Note that if this option is not used, all volumes will be the same (constant).

        For the lognormal distribution, param1 and param2 denote the mean and std of the distribution
        respectively.

        For the uniform distribution, param1 and param2 represent the low and high values respectively.
        """,
        required=False,
    )


def get_save_path_specs(prefix: str) -> dict[str, Any]:
    default_path = f"./{prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.{sd.PropertyExtension.CSV}"
    return dict(
        default=default_path,
        type=click.Path(exists=False),
        show_default=False,
        help=f"""The path to save the generated data. The path must contain the
    supported file extension: {", ".join(sd.PropertyExtension)}.

    For instance:

        ./mydir/myfile.csv

    If this is not provided, it will default to ./{prefix}-[DATE-TIME].csv.
    """,
    )


def validate_box_specs(
    box_specs: tuple[float, float, float] | tuple[float, float],
) -> None:
    box_len = len(box_specs)
    if box_len not in (2, 3):
        raise click.BadArgumentUsage(
            f"BOX_SPECS expects 2 or 3 inputs but {box_len} are given. Please try again."
        )

    return None


def validate_dist_params(dist_params: tuple[str, float, float]) -> None:
    dist, param1, param2 = dist_params

    if dist not in sd.Distribution:
        raise click.UsageError(
            f"value must be one of [{', '.join(sd.Distribution)}] but {dist} is given."
        )

    if not all(isinstance(val, float) for val in (param1, param2)):
        raise click.UsageError(
            f"""
            param1 and param2 are expected to be set to float if {sd.Distribution.UNIFORM} or {sd.Distribution.LOGNORMAL} is chosen;
            but {(param1, param2)} are given. Please try again.
            """,
        )
    return None


@click.command(
    help="""
Sample single phase target volumes using BOX_SPECS.

Args:

    \b
    BOX_SPECS: the specifications of the box.
    The sum of the sampled volumes will be equal to the volume of the box. 

    \b
    If 2D, then the length and  breadth must be given,
    separared by a single space (e.g., 3.0 4.0).

    \b
    If 3D, then the length, breadth, and height must be given separared
    by a single space (e.g., 3.0 4.0 5.0).
"""
)
@click.argument("box-specs", nargs=-1, type=click.FLOAT)
@click.option(
    "--n-grains",
    default=1000,
    type=click.IntRange(min=1, max=None, min_open=False),
    show_default=True,
    help="""
    The number of grains in the domain or box.
    """,
)
@click.option("--dist-params", **get_dist_params_specs())
@click.option("--save-path", **get_save_path_specs("spvols"))
def sample_spvols(
    box_specs: tuple[float, float, float] | tuple[float, float],
    n_grains: int,
    dist_params: tuple[str, float, float] | None,
    save_path: pathlib.Path | str,
) -> None:
    sd.validate_path(save_path, allowed_file_ext=[ext for ext in sd.PropertyExtension])
    validate_box_specs(box_specs)

    click.echo(
        sd.color_text(
            "* Sampling single phase target volumes...",
            color="green",
            bold=True,
        ),
        color=True,
    )

    if dist_params is None:
        click.echo(
            sd.color_text(
                "! Warning: no volume distribution params given, all target volumes will be the same.",
                color="yellow",
                bold=True,
            ),
            color=True,
        )
        volumes = sd.sample_single_phase_vols(
            n_grains=n_grains,
            domain_vol=np.prod(box_specs),
        )
    else:
        validate_dist_params(dist_params)
        volumes = sd.sample_single_phase_vols(
            n_grains=n_grains,
            domain_vol=np.prod(box_specs),
            **generate_dist_kwargs(*dist_params),
        )

    df = pd.DataFrame(
        data=volumes,
        columns=[sd.TARGET_VOLUMES_COLNAME],
    )
    sd.write_df_to_path(
        df=df,
        path=save_path,
    )

    click.echo(
        sd.color_text(
            f"✔ Target volumes sampling done! The sampled target volumes have been saved in {save_path}.",
            color="green",
            bold=True,
        ),
        color=True,
    )

    return None


@click.command(
    help="""
Sample dual phase target volumes using BOX_SPECS.

Args:

    \b
    BOX_SPECS: the specifications of the box. The sum
    of the sampled volumes will be equal to the volume of the box.

    \b
    If 2D, then the length and  breadth must be given,
    separared by a single space (e.g., 3.0 4.0).

    \b
    If 3D, then the length, breadth, and height must be given separared
    by a single space (e.g., 3.0 4.0 5.0).
"""
)
@click.argument("box-specs", nargs=-1, type=click.FLOAT)
@click.option(
    "--n-grains",
    nargs=2,
    default=(500, 500),
    type=click.Tuple(
        [
            click.IntRange(min=1, max=None, min_open=False),
            click.IntRange(min=1, max=None, min_open=False),
        ]
    ),
    show_default=True,
    help="""
    The number of grains in each phase.

    Inputs must be postive integers (>=1) and must follow the following convention:

        --n-grains n1 n2
    """,
)
@click.option(
    "--vol-ratio",
    nargs=2,
    default=(1, 1),
    type=click.Tuple(
        [
            click.FloatRange(min=0, max=None, min_open=True),
            click.FloatRange(min=0, max=None, min_open=True),
        ]
    ),
    show_default=True,
    help="""
    The volume ratio of each phase.

    Inputs must be postive floats (>0) and must follow the following convention:

        --vol-ratio r1 r2

    Note: the ratio will be converted to fractions before calculating the total volume
    for each phase.
    """,
)
@click.option(
    "--dist-params1",
    **get_dist_params_specs("Phase 1 target volumes distribution params."),
)
@click.option(
    "--dist-params2",
    **get_dist_params_specs("Phase 2 target volumes distribution params."),
)
@click.option("--save-path", **get_save_path_specs("dpvols"))
def sample_dpvols(
    box_specs: tuple[float, float, float] | tuple[float, float],
    n_grains: tuple[int, int],
    vol_ratio: tuple[float, float],
    dist_params1: tuple[str, float, float] | None,
    dist_params2: tuple[str, float, float] | None,
    save_path: pathlib.Path | str,
):
    sd.validate_path(save_path, allowed_file_ext=[ext for ext in sd.PropertyExtension])
    validate_box_specs(box_specs)

    click.echo(
        sd.color_text(
            "* Sampling dual phase target volumes...",
            color="green",
            bold=True,
        ),
        color=True,
    )

    dist_kwargs = []
    for i, p in enumerate([dist_params1, dist_params2], start=1):
        if p is None:
            click.echo(
                sd.color_text(
                    f"! Warning: no volume distribution params is provided for phase {i}, all target volumes will be the same.",
                    color="yellow",
                    bold=True,
                ),
                color=True,
            )
            dist_kwargs.append({})
        else:
            validate_dist_params(p)
            dist_kwargs.append(generate_dist_kwargs(*p))

    volumes = sd.sample_dual_phase_vols(
        n_grains=n_grains,
        vol_ratio=vol_ratio,
        domain_vol=np.prod(box_specs),
        dist_kwargs=tuple(dist_kwargs),
    )

    df = pd.DataFrame(
        data=volumes,
        columns=[sd.TARGET_VOLUMES_COLNAME],
    )

    sd.write_df_to_path(
        df=df,
        path=save_path,
    )

    click.echo(
        sd.color_text(
            f"✔ Target volumes sampling done! The sampled target volumes have been saved in {save_path}.",
            color="green",
            bold=True,
        ),
        color=True,
    )

    return None


@click.command(
    help="""
Sample random seeds with BOX_SPECS.

The sampled seeds will be uniforly distributed in

    [0, length) x [0, breadth)

for 2D case, and in

    [0, length) x [0, breadth) x [0, height)

for 3D case.

Args:

    \b
    BOX_SPECS: the specifications of the box.

    \b
    If 2D, then the length and  breadth must be given,
    separared by a single space (e.g., 3.0 4.0).

    \b
    If 3D, then the length, breadth, and height must be given
    separared by a single space (e.g., 3.0 4.0 5.0).
"""
)
@click.argument("box-specs", nargs=-1, type=click.FLOAT)
@click.option(
    "--n-grains",
    default=1000,
    type=click.IntRange(min=1, max=None, min_open=False),
    show_default=True,
    help="""
    The number of grains in the box.
    """,
)
@click.option("--save-path", **get_save_path_specs("seeds"))
def sample_seeds(
    box_specs: tuple[float, float, float] | tuple[float, float],
    n_grains: int,
    save_path: pathlib.Path | str,
) -> None:
    validate_box_specs(box_specs)
    sd.validate_path(save_path, allowed_file_ext=[ext for ext in sd.PropertyExtension])

    click.echo(
        sd.color_text(
            f"* Sampling seeds in the box {' x '.join([str([0, s]) for s in box_specs])}...",
            color="green",
            bold=True,
        ),
        color=True,
    )

    samples = sd.sample_random_seeds(box_specs=box_specs, n_grains=n_grains)

    df = pd.DataFrame(
        data=samples,
        columns=list(sd.COORDINATES)[: samples.shape[1]],
    )
    sd.write_df_to_path(df=df, path=save_path)

    click.echo(
        sd.color_text(
            f"✔ Seeds sampling done! The sampled seeds have been saved in {save_path}.",
            color="green",
            bold=True,
        ),
        color=True,
    )

    return None


@click.command(
    help="""
Generate microstructure with a box with BOX_SPECS, initial
seeds saved in SEEDS_PATH, and target volumes saved in TARGET_VOLS_PATH.

Args:

    \b
    BOX_SPECS: the specifications of the box. If 2D, then the length and breadth
    must be given, separared by a single space (e.g., 3.0 4.0). If 3D, then the
    length, breadth, and height must be given separared by a single space (e.g., 3.0 4.0 5.0).

    \b
    SEEDS_PATH: path to saved intial seed positions. File must be either csv or txt.

    \b
    TARGET_VOLS_PATH: path to saved target volumes. File must be either csv or txt.
"""
)
@click.argument("box-specs", nargs=-1, type=click.FLOAT)
@click.argument(
    "seeds-path",
    type=click.Path(exists=True),
    nargs=1,
)
@click.argument(
    "target-vols-path",
    type=click.Path(exists=True),
    nargs=1,
)
@click.option(
    "--n-iter",
    type=click.IntRange(min=0, max=None, min_open=False),  # FIXME: add max iter
    default=5,
    show_default=True,
    help="""
    The number of iterations of Lloyd's algorithm.
    """,
)
@click.option(
    "--tol",
    type=click.FloatRange(min=0.0, max=None, min_open=True),
    default=0.1,
    show_default=True,
    help="""
    The relative percentange error for volumes.  
    """,
)
@click.option(
    "--damp-param",
    type=click.FloatRange(
        min=0.0,
        max=1.0,
        min_open=False,
        max_open=False,
    ),
    default=1.0,
    show_default=True,
    help="""
    The damping parameter for the Lloyd's algorithm.
    """,
)
@click.option(
    "--periodic",
    is_flag=True,
    help="""
    Give this flag to make the underlying domain periodic in 
    all directions.
    """,
)
@click.option(
    "--verbose",
    is_flag=True,
    help="""
    Give this flag to print out Lloyd's steps.
    """,
)
@click.option(
    "--colorby",
    type=click.Choice([c.value for c in sd.Colorby], case_sensitive=True),
    default=sd.Colorby.FITTED_VOLUMES.value,
    show_default=True,
    help=f"""Specify how to color the generated microstructure.
    Input must be one of [{", ".join(sd.Colorby)}].
        """,
)
@click.option(
    "--colormap",
    type=click.STRING,
    default="plasma",
    show_default=True,
    help="""Specify what colormap to use. Input must be
    a valid matplotlib colormap string 
        """,
)
@click.option(
    "--add-final-seed-positions",
    is_flag=True,
    help="""
    Give this flag to add the final seed positions to 
    the generated diagram. Note that the opacity needs to 
    set to a low value to see the seeds in 3D diagrams.
    """,
)
@click.option(
    "--opacity",
    type=click.FloatRange(
        min=0.0,
        max=1.0,
        min_open=False,
        max_open=False,
    ),
    default=1.0,
    show_default=True,
    help="""
    The opacity of the generated diagram.
    """,
)
@click.option(
    "--window-size",
    type=click.Tuple([int, int]),
    nargs=2,
    default=(600, 600),
    help="The window size (in pixels) of the generated diagram.",
)
@click.option(
    "--save-path",
    type=click.Path(exists=False),
    default=f"./sm-out-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.zip",
    show_default=False,
    help="""
    The path to write all outputs to. The path must contain a file name with 
    extension ".zip".

    Default to sm-out-[DATE]-[TIME].zip.
    """,
)
def generate(
    box_specs: tuple[float, float] | tuple[float, float, float],
    seeds_path: pathlib.Path | str,
    target_vols_path: pathlib.Path | str,
    n_iter: int,
    tol: float,
    damp_param: float,
    periodic: bool,
    verbose: bool,
    colorby: str,
    colormap: str,
    add_final_seed_positions: bool,
    opacity: float,
    window_size: tuple[int, int],
    save_path: pathlib.Path | str,
) -> None:
    validate_box_specs(box_specs)
    for p in (seeds_path, target_vols_path):
        sd.validate_path(p, allowed_file_ext=[ext for ext in sd.PropertyExtension])
    sd.validate_path(save_path, allowed_file_ext=["zip"])

    click.echo(
        sd.color_text(
            "* Generating microstructure...",
            color="green",
            bold=True,
        ),
        color=True,
    )

    space_dim = len(box_specs)
    domain = np.array([[0, s] for s in box_specs])

    seeds = pd.read_csv(seeds_path, index_col=False)
    sd.validate_df(
        seeds,
        expected_colnames=list(sd.COORDINATES)[:space_dim],
        expected_type="float",
        file="seeds",
        bounds=dict(
            zip(
                sd.COORDINATES[:space_dim],
                domain,
            )
        ),
    )

    target_volumes = pd.read_csv(target_vols_path, index_col=False)
    sd.validate_df(
        target_volumes,
        expected_colnames=[sd.TARGET_VOLUMES_COLNAME],
        expected_type="float",
        file="volumes",
    )

    if seeds.shape[0] != target_volumes.shape[0]:
        raise click.UsageError(
            f"""the number of samples in seeds and target volumes don't match:
            number samples in seeds={seeds.shape[0]}, number of samples in target volumes={target_volumes.shape[0]}.
            """
        )

    # get the underlying arrays in seeds and volumes
    seeds = seeds.values
    target_volumes = target_volumes[sd.TARGET_VOLUMES_COLNAME].values

    # check the total target volumes match the domain volume
    VOL_DIFF_TOL = 1e-6
    domain_vol = np.prod(box_specs)
    diff = abs(target_volumes.sum() - domain_vol)
    if diff > VOL_DIFF_TOL:
        raise click.UsageError(
            f"""Mismatch total volume: domain volume is {domain_vol:.2f} whereas total uploaded volume is {target_volumes.sum():.2f};
            a difference of {diff:.2f}. Volume difference must be at most {VOL_DIFF_TOL:.2e}."""
        )

    diagram = sd.generate_diagram(
        domain=domain,
        seeds=seeds,
        volumes=target_volumes,
        periodic=periodic,
        tol=tol,
        n_iter=n_iter,
        damp_param=damp_param,
        verbose=verbose,
    )

    mesh, plotter = sd.plot_diagram(
        generator=diagram.generator,
        target_volumes=target_volumes,
        colorby=colorby,
        colormap=colormap,
        window_size=window_size,
        add_final_seed_positions=add_final_seed_positions,
        opacity=opacity,
    )

    sd.save_results_as_zip(
        diagram=diagram,
        mesh=mesh,
        plotter=plotter,
        path=save_path,
    )

    click.echo(
        "\n✔ Fit summary: \n"
        f"\t- mean percentage error = {diagram.mean_percentage_error:.4f}%\n"
        f"\t- max percentage error = {diagram.max_percentage_error:.4f}%"
    )

    click.echo(
        sd.color_text(
            f"✔ Microstructure generation done! Results are saved in {save_path}.",
            color="green",
            bold=True,
        ),
        color=True,
    )
