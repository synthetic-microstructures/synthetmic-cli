import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Iterable

import click
import numpy as np
import pandas as pd
import pyvista as pv
from synthetmic import LaguerreDiagramGenerator

COORDINATES = ("x", "y", "z")
TARGET_VOLUMES_COLNAME = "target_volumes"


@dataclass(frozen=True)
class Diagram:
    generator: LaguerreDiagramGenerator
    max_percentage_error: float
    mean_percentage_error: float
    centroids: np.ndarray
    vertices: dict[int, list]
    fitted_volumes: np.ndarray
    target_volumes: np.ndarray
    weights: np.ndarray
    seeds: np.ndarray
    positions: np.ndarray
    domain: np.ndarray


class Distribution(StrEnum):
    UNIFORM = auto()
    LOGNORMAL = auto()


class FigureExtension(StrEnum):
    PDF = auto()
    SVG = auto()
    EPS = auto()
    HTML = auto()
    VTK = auto()


class PropertyExtension(StrEnum):
    CSV = auto()
    TXT = auto()


class Colorby(StrEnum):
    TARGET_VOLUMES = auto()
    FITTED_VOLUMES = auto()
    VOLUME_ERRORS = auto()
    RANDOM = auto()


def color_text(text: str, color: str = "green", bold: bool = False) -> str:
    return click.style(text=text, fg=color, bold=bold)


def sample_single_phase_vols(
    n_grains: int,
    domain_vol: float,
    **kwargs: dict[str, float],
) -> np.ndarray:
    if kwargs:
        if {"low", "high"}.issubset(kwargs):
            samples = np.random.uniform(
                low=kwargs.get("low"), high=kwargs.get("high"), size=n_grains
            )
            scaling_factor = domain_vol / samples.sum()

            return scaling_factor * samples

        if {"mean", "std"}.issubset(kwargs):
            sigma = np.sqrt(np.log(1 + (kwargs.get("std") / kwargs.get("mean")) ** 2))
            mu = -0.5 * sigma**2 + np.log(kwargs.get("mean"))

            samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_grains)
            scaling_factor = domain_vol / samples.sum()

            return scaling_factor * samples

    rel_vol = np.ones(n_grains) / n_grains

    return rel_vol * domain_vol


def sample_dual_phase_vols(
    n_grains: tuple[int, int],
    vol_ratio: tuple[float, float],
    domain_vol: float,
    dist_kwargs: tuple[dict[str, float], dict[str, float]],
) -> np.ndarray:
    phase1_vol = (vol_ratio[0] / sum(vol_ratio)) * domain_vol
    phase2_vol = domain_vol - phase1_vol

    domain_vols = (phase1_vol, phase2_vol)

    return np.concatenate(
        tuple(
            [
                sample_single_phase_vols(
                    n_grains=n_grains[i],
                    domain_vol=domain_vols[i],
                    **dist_kwargs[i],
                )
                for i in (0, 1)
            ]
        ),
        axis=None,
    )


def sample_random_seeds(
    box_specs: Iterable[float],
    n_grains: int,
    random_state: int | None = None,
) -> np.ndarray:
    np.random.seed(random_state)

    seeds = np.column_stack(
        [np.random.uniform(low=0, high=h, size=n_grains) for h in box_specs]
    )

    return seeds


def write_df_to_path(
    df: pd.DataFrame,
    path: str,
) -> None:
    ext = path.split(".")[-1]
    if ext not in PropertyExtension:
        raise ValueError(
            f"Wrong file extension: {ext}. Extension must be one of [{', '.join(PropertyExtension)}]."
        )

    df.to_csv(path, index=False)

    return None


def describe_df(df: pd.DataFrame) -> None:
    print("\n✔ Shape:")
    print(f"\t- number of rows: {df.shape[0]:,}")
    print(f"\t- number of columns: {df.shape[1]:,}")

    print("\n✔ Column names:")
    print(f"\t- columns: {list(df.columns)}")

    print("\n✔ Detailed column information:")
    print("\t" + "-" * 60)
    print(
        f"\t{'column name':<14} {'data type':<12} {'non-null':<12} {'null':<12} {'% null':<6}"
    )
    print("\t" + "-" * 60)

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_count = df[col].count()
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / len(df)) * 100

        print(
            f"\t{col:<14} {dtype:<12} {non_null_count:<12} {null_count:<12} {null_percentage:<6.1f}",
        )

    print("\t" + "-" * 60)

    total_values = df.size
    total_non_null = df.count().sum()
    total_null = df.isnull().sum().sum()

    print("\n✔ Summary:")
    print(f"\t- total values: {total_values:,}")
    print(f"\t- total non-null values: {total_non_null:,}")
    print(f"\t- total null values: {total_null:,}")
    print(f"\t- overall null percentage: {(total_null / total_values) * 100:.1f}%")

    print("\n✔ Data types summary:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"\t- {dtype}: {count} column(s)")

    return None


def generate_diagram(
    domain: np.ndarray,
    seeds: np.ndarray,
    volumes: np.ndarray,
    periodic: bool,
    tol: float,
    n_iter: int,
    damp_param: float,
    verbose: bool,
) -> Diagram:
    generator = LaguerreDiagramGenerator(
        tol=tol,
        n_iter=n_iter,
        damp_param=damp_param,
        verbose=verbose,
    )
    generator.fit(
        seeds=seeds,
        volumes=volumes,
        domain=domain,
        periodic=[True] * domain.shape[0] if periodic else None,
        init_weights=None,
    )

    return Diagram(
        generator=generator,
        max_percentage_error=generator.max_percentage_error_,
        mean_percentage_error=generator.mean_percentage_error_,
        centroids=generator.get_centroids(),
        vertices=generator.get_vertices(),
        target_volumes=volumes,
        fitted_volumes=generator.get_fitted_volumes(),
        weights=generator.get_weights(),
        domain=domain,
        seeds=seeds,
        positions=generator.get_positions(),
    )


def plot_diagram(
    generator: LaguerreDiagramGenerator,
    target_volumes: np.ndarray,
    colorby: str,
    colormap: str = "plasma",
    window_size: tuple[int, int] = (400, 400),
    add_final_seed_positions: bool = False,
    opacity: float = 1.0,
) -> tuple[pv.PolyData | pv.UnstructuredGrid, dict[str, pv.Plotter]]:
    mesh = generator.get_mesh()

    match colorby:
        case Colorby.TARGET_VOLUMES:
            colorby_values = target_volumes

        case Colorby.FITTED_VOLUMES:
            colorby_values = generator.get_fitted_volumes()

        case Colorby.VOLUME_ERRORS:
            colorby_values = (
                np.abs(generator.get_fitted_volumes() - target_volumes)
                * 100
                / target_volumes
            )

        case Colorby.RANDOM:
            colorby_values = np.random.rand(target_volumes.shape[0])

        case _:
            raise ValueError(
                f"Invalid colorby: {colorby}. Value must be one of {', '.join(Colorby)}"
            )

    mesh.cell_data["vols"] = colorby_values[mesh.cell_data["num"].astype(int)]

    pl = pv.Plotter(
        off_screen=True,
        window_size=list(window_size),
    )

    pl.add_mesh(
        mesh,
        show_edges=True,
        show_scalar_bar=False,
        lighting=False,
        cmap=colormap,
        opacity=opacity,
    )

    final_seed_positions = generator.get_positions()
    n_samples, space_dim = final_seed_positions.shape

    if space_dim == 2:
        pl.camera_position = "xy"

        final_seed_positions = np.column_stack(
            (final_seed_positions, np.zeros(n_samples))
        )
    elif space_dim == 3:
        pl.camera_position = "yz"
        pl.camera.azimuth = 45

    if add_final_seed_positions:
        pl.add_points(
            points=final_seed_positions,
            render_points_as_spheres=True,
            color="black",
            point_size=5,
        )

    pl.show_axes()

    return mesh, pl


def validate_df(
    df: pd.DataFrame,
    expected_colnames: list[str],
    expected_type: str,
    file: str,
    expected_dim: tuple[int, int] | None = None,
    bounds: dict[str, Iterable[float]] | None = None,
) -> None:
    df_colnames = df.columns.to_list()

    if df_colnames != expected_colnames:
        raise click.UsageError(
            f"Column mismatch error in the uploaded {file} file: expected {expected_colnames} but got {df_colnames}. Please try again."
        )

    if expected_dim is not None:
        if df.shape != expected_dim:
            raise click.UsageError(
                f"Dimension mismatch error in the uploaded {file} file: expected dimension {expected_dim} but got {df.shape}. Please try again."
            )

    df_dtypes = df.dtypes.values.tolist()
    if not all(t == expected_type for t in df_dtypes):
        raise click.UsageError(
            f"Data type mismatch error in the uploaded {file} file: expected all values to be of {expected_type} but got {df_dtypes}. Please try again."
        )

    if bounds is not None:
        msg = []
        for c, b in bounds.items():
            c_min = df[c].values.min()
            c_max = df[c].values.max()

            if not all(b[0] <= val <= b[1] for val in [c_min, c_max]):
                msg.append(
                    f"""expected {c}-coordinate values to be in [{b[0]:.2f}, {b[1]:.2f}]
                but values are in [{c_min:.2f}, {c_max:.2f}]"""
                )

        if msg:
            raise click.UsageError(
                f"Value bound error in the uploaded {file} file: "
                + "; ".join(msg)
                + ". Please try again."
            )

    return None


def validate_path(path: str, allowed_file_ext: list[str]) -> None:
    ext = path.split(".")[-1]
    if ext not in allowed_file_ext:
        raise click.UsageError(
            f"Invalid path: path must contain a file with extension in {allowed_file_ext} but got {ext}."
        )

    return None


def extract_property_as_df(diagram: Diagram) -> dict[str, pd.DataFrame]:
    dim = diagram.seeds.shape[1]
    property_dict = {}

    property_dict["domain"] = pd.DataFrame(
        data=diagram.domain,
        columns=["a", "b"],
    )

    for p, d in zip(
        (
            "seeds_initial",
            "seeds_final",
            "centroids",
        ),
        (
            diagram.seeds,
            diagram.positions,
            diagram.centroids,
        ),
    ):
        property_dict[p] = pd.DataFrame(data=d[:, :dim], columns=COORDINATES[:dim])

    for p, d in zip(
        (
            "weights",
            "target_volumes",
            "fitted_volumes",
        ),
        (diagram.weights, diagram.target_volumes, diagram.fitted_volumes),
    ):
        property_dict[p] = pd.DataFrame(
            data=d, columns=["weights"] if p == "weights" else ["volumes"]
        )

    return property_dict


def save_results_as_zip(
    diagram: Diagram,
    mesh: pv.PolyData | pv.UnstructuredGrid,
    plotter: pv.Plotter,
    path: str,
) -> None:
    validate_path(path, allowed_file_ext=["zip"])

    diagram_prop = extract_property_as_df(diagram)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname, df in diagram_prop.items():
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            for ext in PropertyExtension:
                zipf.writestr(f"{fname}.{ext}", buffer.getvalue())

            buffer.close()

        # write the vertices to json
        buffer = io.StringIO()
        json.dump(
            diagram.vertices,
            buffer,
            indent=4,
        )
        buffer.seek(0)
        zipf.writestr("vertices.json", buffer.getvalue())
        buffer.close()

        for ext in FigureExtension:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f".{ext}", delete=True
            ) as tmp_file:
                filename = tmp_file.name

                match ext:
                    case (
                        FigureExtension.PDF | FigureExtension.EPS | FigureExtension.SVG
                    ):
                        plotter.save_graphic(filename)

                    case FigureExtension.HTML:
                        plotter.export_html(filename)

                    case FigureExtension.VTK:
                        mesh.save(filename, binary=False)

                with open(filename, "rb") as f:
                    content = f.read()

            zipf.writestr(f"diagram.{ext}", content)

    return None
