#!/usr/bin/env python3
"""Generate simulation-ready CT55 hearts with an implicit planar basal cut.

For every sample this script:

1. samples the trained aggregate posterior and decodes one physical 1404 x 3
   LDDMM momentum field;
2. shoots the CT55 surface and tetrahedral volume without changing the decoded
   momentum field;
3. searches sample-specific planes parallel to the CT55 basal plane;
4. evaluates the analytic affine level set phi(x) = s * (n.x - d) on the
   original valid shot volume and asks MMG3D to insert the zero isosurface;
5. freezes the non-basal anatomy while leaving a narrow cap-wall seam editable,
   keeps only the retained level-set subdomain, and optionally performs local
   MMG3D quality-polishing passes;
6. reprojects only the MMG-created cap vertices exactly onto the selected plane
   and selects the shallowest candidate that passes all verified QA checks; and
7. writes one momentum NPZ, boundary-surface VTK, volume VTU, and JSON QA report.

The final VTK surface is extracted from the accepted VTU, so the two meshes are
geometrically consistent.  Direct tetrahedron clipping is deliberately avoided:
MMG3D receives the valid, uncut shot volume and constructs a conforming planar
interface itself.  Required MMG entities protect the non-basal anatomy, while a
small wall band is released so the cap-wall junction can be retriangulated.  No
scaling, anatomical smoothing, or momentum refinement is performed. Rejection
is explicit: a mesh is never called simulation-ready by silently lowering the
requested quality limits.

Requires: numpy, torch, vtk, and the MMG3D executable.
Install MMG with: conda install -c conda-forge mmgsuite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Use the same Python environment as registration."
    ) from exc

try:
    import vtk
    from vtk.util.numpy_support import (
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
        vtk_to_numpy,
    )
except ImportError as exc:
    raise SystemExit("VTK is required: python3 -m pip install vtk") from exc


GENERATION_VERSION = "ct55_graph_vae_lddmm_mmg_levelset_generation_v4"
REFINEMENT_VERSION = "ct55_parallel_sample_plane_endpoint_refinement_v3"
CLIPPING_VERSION = "mmg3d_affine_levelset_protected_boundary_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "06.1_CT55GraphVAE_L16_H96/ct55_graph_momenta_vae_all61.pt"
        ),
    )
    parser.add_argument(
        "--momenta-dir",
        type=Path,
        default=Path("05.2_Momenta"),
        help=(
            "The 61 training momentum NPZ files. They are encoded to construct "
            "the learned aggregate-posterior sampling distribution."
        ),
    )
    parser.add_argument(
        "--template-prepared",
        type=Path,
        default=Path(
            "05.1_CTRegistered/RegistrationMetadata/Prepared/ct_case_0055.npz"
        ),
        help="Optional prepared CT55 NPZ; labeled-surface fallback is automatic.",
    )
    parser.add_argument(
        "--template-surface",
        type=Path,
        default=Path("04.1_CTRegistrationSurfaces/ct_case_0055.vtk"),
        help="Original CT55 registration surface and geometry.",
    )
    parser.add_argument(
        "--template-labeled-surface",
        type=Path,
        default=Path("05.1_CTRegistered/ct_case_registered_0055.vtk"),
        help=(
            "Registered CT55 surface carrying BasalFace and AnatomicalObject "
            "cell labels; used only when the prepared NPZ is unavailable."
        ),
    )
    parser.add_argument(
        "--template-volume",
        type=Path,
        default=Path("03.1_ICPAlignedCT55/ct_case_0055.vtu"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("07.1_Generated"),
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--sampling-policy",
        choices=("aggregate-posterior", "standard-normal"),
        default="aggregate-posterior",
        help=(
            "aggregate-posterior samples the learned mixture of 61 subject "
            "posteriors; standard-normal is retained only for controlled tests."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Latent expansion about the aggregate-posterior mean. Use 1.0 for "
            "the fitted distribution. For standard-normal, this is its SD."
        ),
    )
    parser.add_argument(
        "--maximum-standardized-latent-radius",
        type=float,
        default=4.5,
        help=(
            "Euclidean radius guard for standard-normal draws before applying "
            "temperature. It is unused by aggregate-posterior sampling."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted flat-folder run. Complete samples are kept; "
            "partial samples cause an error instead of being overwritten."
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--maximum-plane-offset-shift",
        type=float,
        default=30.0,
        help=(
            "Hard absolute bound, in mm from CT55's plane, for the automatic "
            "sample-specific cut. This is a safety bound, not a target height."
        ),
    )
    parser.add_argument(
        "--cut-clearance",
        type=float,
        default=0.10,
        help="Initial inward clearance beyond every deformed old-basal vertex (mm).",
    )
    parser.add_argument(
        "--cut-search-distance",
        type=float,
        default=25.0,
        help="Maximum additional inward distance explored from the complete cut (mm).",
    )
    parser.add_argument(
        "--cut-search-step",
        type=float,
        default=0.5,
        help="Nominal spacing of candidate plane offsets (mm).",
    )
    parser.add_argument(
        "--cut-search-maximum-candidates",
        type=int,
        default=128,
        help="Maximum surface-cut candidates constructed before pre-screening.",
    )
    parser.add_argument(
        "--remesh-candidates",
        type=int,
        default=20,
        help=(
            "Maximum topology-valid planes sent to MMG3D level-set insertion "
            "per generated heart. Candidates are tried from shallowest to "
            "deepest and the search stops at the first passing mesh."
        ),
    )
    parser.add_argument(
        "--expected-basal-contours",
        type=int,
        default=3,
        help=(
            "Required cap-boundary loops: outer wall, LV opening, RV opening. "
            "Use 0 only to disable this anatomy check."
        ),
    )
    parser.add_argument(
        "--flat-tolerance",
        type=float,
        default=1e-6,
        help="Maximum final basal-plane residual in mm.",
    )
    parser.add_argument(
        "--minimum-scaled-jacobian",
        type=float,
        default=0.02,
        help="Required minimum VTK tetrahedral scaled Jacobian.",
    )
    parser.add_argument(
        "--minimum-p01-scaled-jacobian",
        type=float,
        default=0.10,
        help="Required first-percentile tetrahedral scaled Jacobian.",
    )
    parser.add_argument(
        "--minimum-dihedral-angle",
        type=float,
        default=2.0,
        help="Required minimum tetrahedral angle in degrees.",
    )
    parser.add_argument(
        "--collar-width",
        type=float,
        default=8.0,
        help=(
            "Geometric depth in mm of the editable basal level-set collar. "
            "Retained tetrahedra deeper than the collar are written as "
            "RequiredTetrahedra for MMG3D."
        ),
    )
    parser.add_argument(
        "--collar-rings",
        type=int,
        default=3,
        help=(
            "Additional face-adjacency rings grown from the cut cap into the "
            "editable collar."
        ),
    )
    parser.add_argument(
        "--wall-release-width",
        type=float,
        default=0.0,
        help=(
            "Editable anatomical-wall width next to the new cap in mm. Zero "
            "selects an edge-length-aware value automatically. Wall triangles "
            "beyond this band are required MMG entities."
        ),
    )
    parser.add_argument(
        "--mmg3d-executable",
        default="auto",
        help=(
            "MMG3D executable or path. 'auto' searches for mmg3d_O3 and mmg3d."
        ),
    )
    parser.add_argument(
        "--mmg-hausdorff",
        type=float,
        default=0.01,
        help=(
            "MMG boundary approximation tolerance in mm. The anatomical wall is "
            "also marked required, so this mainly controls the planar cap."
        ),
    )
    parser.add_argument(
        "--mmg-gradation",
        type=float,
        default=1.3,
        help="Maximum MMG edge-size gradation.",
    )
    parser.add_argument(
        "--mmg-timeout-seconds",
        type=float,
        default=600.0,
        help=(
            "Hard wall-clock limit for each isolated MMG3D attempt. A timeout "
            "rejects only that attempt and the plane search continues."
        ),
    )
    parser.add_argument(
        "--mmg-quality-passes",
        type=int,
        default=2,
        help=(
            "Maximum additional local MMG -optim passes after level-set "
            "insertion. Zero disables post-level-set polishing."
        ),
    )
    parser.add_argument(
        "--no-mmg-nofem-fallback",
        dest="mmg_nofem_fallback",
        action="store_false",
        help=(
            "Disable the conservative second MMG3D attempt using -nofem when "
            "the standard optimization fails."
        ),
    )
    parser.set_defaults(mmg_nofem_fallback=True)
    parser.add_argument("--shooting-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    integer_positive = (
        "count",
        "threads",
        "cut_search_maximum_candidates",
        "remesh_candidates",
        "collar_rings",
        "shooting_chunk_size",
    )
    for name in integer_positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if (
        not math.isfinite(args.maximum_standardized_latent_radius)
        or args.maximum_standardized_latent_radius <= 0
    ):
        parser.error("--maximum-standardized-latent-radius must be positive")
    positive = (
        "maximum_plane_offset_shift",
        "cut_clearance",
        "cut_search_distance",
        "cut_search_step",
        "flat_tolerance",
        "minimum_scaled_jacobian",
        "minimum_p01_scaled_jacobian",
        "minimum_dihedral_angle",
        "collar_width",
        "mmg_hausdorff",
        "mmg_gradation",
        "mmg_timeout_seconds",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.expected_basal_contours < 0:
        parser.error("--expected-basal-contours must be nonnegative")
    if args.mmg_quality_passes < 0:
        parser.error("--mmg-quality-passes must be nonnegative")
    if not math.isfinite(args.wall_release_width) or args.wall_release_width < 0:
        parser.error("--wall-release-width must be finite and nonnegative")
    if args.minimum_scaled_jacobian > 1:
        parser.error("--minimum-scaled-jacobian cannot exceed 1")
    if args.minimum_p01_scaled_jacobian > 1:
        parser.error("--minimum-p01-scaled-jacobian cannot exceed 1")
    if not 0 < args.minimum_dihedral_angle < 90:
        parser.error("--minimum-dihedral-angle must lie between 0 and 90 degrees")
    if args.mmg_gradation <= 1:
        parser.error("--mmg-gradation must be greater than 1")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, neighbors: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("neighbor_indices", neighbors)
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        neighboring = features[:, self.neighbor_indices, :].mean(dim=2)
        return self.self_linear(features) + self.neighbor_linear(neighboring)


class ResidualGraphBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        neighbors: torch.Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        self.graph = GraphConvolution(input_dim, output_dim, neighbors)
        self.normalization = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim, bias=False)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        update = torch.nn.functional.gelu(self.normalization(self.graph(features)))
        return self.skip(features) + self.dropout(update)


class GraphMomentaBetaVAE(nn.Module):
    """Exact architecture saved by 06_fit_ct55_graph_vae_all.py."""

    def __init__(
        self,
        coordinates: torch.Tensor,
        neighbors: torch.Tensor,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout_probability = dropout
        self.register_buffer("coordinates", coordinates)
        self.encoder_1 = ResidualGraphBlock(6, hidden_dim, neighbors, dropout)
        self.encoder_2 = ResidualGraphBlock(hidden_dim, hidden_dim, neighbors, dropout)
        self.encoder_3 = ResidualGraphBlock(hidden_dim, hidden_dim, neighbors, dropout)
        self.encoder_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_log_variance = nn.Linear(hidden_dim, latent_dim)
        positional_dim = 3 + 3 * 2 * 3
        self.decoder_input = nn.Sequential(
            nn.Linear(positional_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.decoder_1 = ResidualGraphBlock(hidden_dim, hidden_dim, neighbors, dropout)
        self.decoder_2 = ResidualGraphBlock(hidden_dim, hidden_dim, neighbors, dropout)
        self.decoder_output = nn.Linear(hidden_dim, 3)

    def positional_encoding(self) -> torch.Tensor:
        components = [self.coordinates]
        for frequency in (1.0, 2.0, 4.0):
            argument = math.pi * frequency * self.coordinates
            components.extend((torch.sin(argument), torch.cos(argument)))
        return torch.cat(components, dim=-1)

    def encode(
        self, normalized_momenta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = normalized_momenta.shape[0]
        coordinates = self.coordinates.unsqueeze(0).expand(batch, -1, -1)
        hidden = torch.cat((coordinates, normalized_momenta), dim=-1)
        hidden = self.encoder_1(hidden)
        hidden = self.encoder_2(hidden)
        hidden = self.encoder_3(hidden)
        pooled = torch.cat((hidden.mean(dim=1), hidden.amax(dim=1)), dim=-1)
        encoded = self.encoder_head(pooled)
        mu = self.to_mu(encoded)
        log_variance = torch.clamp(self.to_log_variance(encoded), -10.0, 6.0)
        return mu, log_variance

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        positions = self.positional_encoding().unsqueeze(0).expand(batch, -1, -1)
        latent_field = latent[:, None, :].expand(-1, positions.shape[1], -1)
        hidden = self.decoder_input(torch.cat((positions, latent_field), dim=-1))
        hidden = self.decoder_1(hidden)
        hidden = self.decoder_2(hidden)
        return self.decoder_output(hidden)


def checkpoint_value(checkpoint: dict[str, Any], name: str) -> Any:
    if name not in checkpoint:
        raise ValueError(f"Checkpoint is missing {name!r}")
    return checkpoint[name]


def sample_latents(
    count: int,
    latent_dim: int,
    temperature: float,
    maximum_radius: float,
    seed: int,
) -> np.ndarray:
    """Antithetic truncated-normal draws; every untruncated marginal is N(0,T²)."""
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    attempts = 0
    while len(accepted) < count:
        candidate = rng.standard_normal(latent_dim)
        attempts += 1
        if np.linalg.norm(candidate) > maximum_radius:
            if attempts > 100000:
                raise RuntimeError("Could not draw latent vectors within the radius guard")
            continue
        accepted.append(temperature * candidate)
        if len(accepted) < count:
            accepted.append(-temperature * candidate)
    result = np.stack(accepted[:count]).astype(np.float32)
    if result.shape != (count, latent_dim) or not np.isfinite(result).all():
        raise RuntimeError("Latent sampling produced an invalid array")
    return result


def load_training_momenta(
    directory: Path, expected_control: np.ndarray, expected_count: int = 61
) -> tuple[np.ndarray, list[str]]:
    """Load the fitted cohort while preserving the shared control-point order."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Momenta directory not found: {directory}")
    momenta: list[np.ndarray] = []
    names: list[str] = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "momenta" not in data:
                continue
            value = np.asarray(data["momenta"], dtype=np.float64)
            if value.shape != expected_control.shape or not np.isfinite(value).all():
                raise ValueError(f"{path}: invalid momentum array {value.shape}")
            if "control_points" in data:
                saved_control = np.asarray(data["control_points"], dtype=np.float64)
                if saved_control.shape != expected_control.shape:
                    raise ValueError(f"{path}: control-point shape differs")
                maximum_error = float(
                    np.max(np.abs(saved_control - expected_control))
                )
                if maximum_error > 5e-5:
                    raise ValueError(
                        f"{path}: control-point grid differs from checkpoint "
                        f"(maximum coordinate error={maximum_error:.6g})"
                    )
            momenta.append(value)
            names.append(path.stem)
    if len(momenta) != expected_count:
        raise ValueError(
            f"Expected {expected_count} training momentum files, found {len(momenta)}"
        )
    return np.stack(momenta), names


def sample_aggregate_posterior(
    model: GraphMomentaBetaVAE,
    training_momenta: np.ndarray,
    mean_field: np.ndarray,
    momentum_scale: float,
    count: int,
    temperature: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Sample q(z)=N^-1 sum_i q(z|x_i), optionally expanded about its mean."""
    normalized = ((training_momenta - mean_field) / momentum_scale).astype(
        np.float32
    )
    with torch.inference_mode():
        mu_tensor, log_variance_tensor = model.encode(
            torch.from_numpy(normalized).to(device)
        )
    mu = mu_tensor.cpu().numpy().astype(np.float64)
    log_variance = log_variance_tensor.cpu().numpy().astype(np.float64)
    if not np.isfinite(mu).all() or not np.isfinite(log_variance).all():
        raise FloatingPointError("Encoder produced non-finite posterior parameters")
    rng = np.random.default_rng(seed)
    component = rng.integers(0, len(mu), size=count)
    epsilon = rng.standard_normal((count, mu.shape[1]))
    draw = mu[component] + epsilon * np.exp(0.5 * log_variance[component])
    aggregate_mean = mu.mean(axis=0, keepdims=True)
    latent = aggregate_mean + temperature * (draw - aggregate_mean)
    if not np.isfinite(latent).all():
        raise FloatingPointError("Aggregate-posterior sampling produced non-finite values")
    return latent.astype(np.float32), mu, component.tolist()


def decode_and_save_momenta(
    args: argparse.Namespace,
    template: dict[str, Any],
    output_dir: Path,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dictionary")
    if checkpoint_value(checkpoint, "model_class") != "GraphMomentaBetaVAE":
        raise ValueError("Checkpoint model class is not GraphMomentaBetaVAE")
    if checkpoint_value(checkpoint, "template") != "ct_case_0055":
        raise ValueError("Checkpoint was not trained from CT55")
    if int(checkpoint_value(checkpoint, "fit_subject_count")) != 61:
        raise ValueError("Checkpoint was not fitted on all 61 subjects")

    control = np.asarray(checkpoint_value(checkpoint, "control_points"), dtype=np.float64)
    neighbors = np.asarray(checkpoint_value(checkpoint, "neighbor_indices"), dtype=np.int64)
    coordinate_center = np.asarray(
        checkpoint_value(checkpoint, "coordinate_center"), dtype=np.float64
    )
    coordinate_scale = float(checkpoint_value(checkpoint, "coordinate_scale"))
    mean_field = np.asarray(
        checkpoint_value(checkpoint, "mean_momentum_field"), dtype=np.float64
    )
    momentum_scale = float(checkpoint_value(checkpoint, "momentum_scale"))
    origin = np.asarray(checkpoint_value(checkpoint, "origin"), dtype=np.float64)
    shooting_scale = float(checkpoint_value(checkpoint, "shooting_scale"))
    shooting_steps = int(checkpoint_value(checkpoint, "shooting_steps"))
    latent_dim = int(checkpoint_value(checkpoint, "latent_dim"))
    hidden_dim = int(checkpoint_value(checkpoint, "hidden_dim"))
    dropout = float(checkpoint_value(checkpoint, "dropout"))
    if latent_dim < 1 or control.shape != (1404, 3) or mean_field.shape != control.shape:
        raise ValueError(
            f"Expected a positive latent dimension and 1404x3 fields; got {latent_dim}, "
            f"{control.shape}, {mean_field.shape}"
        )
    arrays = (control, coordinate_center, mean_field, origin)
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Checkpoint geometry or normalization contains non-finite values")
    if coordinate_scale <= 0 or momentum_scale <= 0 or shooting_scale <= 0:
        raise ValueError("Checkpoint contains an invalid scale")
    normalized_control = ((control - coordinate_center) / coordinate_scale).astype(
        np.float32
    )
    model = GraphMomentaBetaVAE(
        torch.from_numpy(normalized_control),
        torch.from_numpy(neighbors),
        latent_dim,
        hidden_dim,
        dropout,
    )
    model.load_state_dict(checkpoint_value(checkpoint, "model_state"), strict=True)
    device = choose_device(args.device)
    model.to(device)
    model.eval()
    posterior_components: list[int | None]
    training_names: list[str] = []
    if args.sampling_policy == "aggregate-posterior":
        training_momenta, training_names = load_training_momenta(
            args.momenta_dir, control, expected_count=61
        )
        latent, _, selected_components = sample_aggregate_posterior(
            model,
            training_momenta,
            mean_field,
            momentum_scale,
            args.count,
            args.temperature,
            args.seed,
            device,
        )
        posterior_components = [int(value) for value in selected_components]
    else:
        latent = sample_latents(
            args.count,
            latent_dim,
            args.temperature,
            args.maximum_standardized_latent_radius,
            args.seed,
        )
        posterior_components = [None] * args.count
    with torch.inference_mode():
        decoded_normalized = model.decode(
            torch.from_numpy(latent).to(device)
        ).cpu().numpy()
    momenta = decoded_normalized.astype(np.float64) * momentum_scale + mean_field
    if momenta.shape != (args.count, 1404, 3) or not np.isfinite(momenta).all():
        raise FloatingPointError("Decoder produced invalid momentum fields")

    samples: list[tuple[int, dict[str, Any]]] = []
    checkpoint_hash = sha256(args.checkpoint)
    for index in range(1, args.count + 1):
        samples.append(
            (
                index,
                {
                    "control_points": control,
                    "momenta": momenta[index - 1],
                    "origin": origin,
                    "scale": shooting_scale,
                    "steps": shooting_steps,
                    "plane_normal": np.asarray(
                        template["plane_normal"], dtype=np.float64
                    ),
                    "plane_offset": float(template["plane_offset"]),
                    "template": "ct_case_0055",
                    "target": f"generated_{index:04d}",
                    "latent": latent[index - 1],
                    "temperature": float(args.temperature),
                    "sampling_policy": args.sampling_policy,
                    "posterior_component_index": posterior_components[index - 1],
                    "posterior_component_source": (
                        training_names[posterior_components[index - 1]]
                        if posterior_components[index - 1] is not None
                        else None
                    ),
                    "generation_version": GENERATION_VERSION,
                    "checkpoint_sha256": checkpoint_hash,
                },
            )
        )
    diagnostics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "count": args.count,
        "latent_dim": latent_dim,
        "temperature": args.temperature,
        "maximum_standardized_latent_radius": args.maximum_standardized_latent_radius,
        "latent_norm_minimum": float(np.linalg.norm(latent, axis=1).min()),
        "latent_norm_median": float(np.median(np.linalg.norm(latent, axis=1))),
        "latent_norm_maximum": float(np.linalg.norm(latent, axis=1).max()),
        "sampling": args.sampling_policy,
        "decoder_device": str(device),
        "momenta_dir": (
            str(args.momenta_dir)
            if args.sampling_policy == "aggregate-posterior"
            else None
        ),
    }
    return samples, diagnostics


def tensor_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def tensor(
    value: np.ndarray | float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def torch_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.exp(-((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))


def control_rhs(
    control: torch.Tensor, momentum: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = control[:, None, :] - control[None, :, :]
    matrix = torch.exp(-(delta**2).sum(-1))
    d_control = matrix @ momentum
    d_momentum = (
        2.0
        * ((matrix * (momentum @ momentum.T))[:, :, None] * delta).sum(1)
    )
    return d_control, d_momentum


def hamiltonian_rhs(
    control: torch.Tensor, momentum: torch.Tensor, shape: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_control, d_momentum = control_rhs(control, momentum)
    d_shape = kernel(shape, control) @ momentum
    return d_control, d_momentum, d_shape


def shoot(
    control: torch.Tensor,
    momentum: torch.Tensor,
    shape: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    dt = 1.0 / steps
    for _ in range(steps):
        dc, dm, ds = hamiltonian_rhs(control, momentum, shape)
        dc2, dm2, ds2 = hamiltonian_rhs(
            control + dt * dc / 2.0,
            momentum + dt * dm / 2.0,
            shape + dt * ds / 2.0,
        )
        control = control + dt * dc2
        momentum = momentum + dt * dm2
        shape = shape + dt * ds2
    return shape


def control_trajectory(
    control: torch.Tensor, momentum: torch.Tensor, steps: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    dt = 1.0 / steps
    trajectory = []
    for _ in range(steps):
        dc, dm = control_rhs(control, momentum)
        midpoint_control = control + dt * dc / 2.0
        midpoint_momentum = momentum + dt * dm / 2.0
        dc2, dm2 = control_rhs(midpoint_control, midpoint_momentum)
        trajectory.append(
            (control, momentum, midpoint_control, midpoint_momentum)
        )
        control = control + dt * dc2
        momentum = momentum + dt * dm2
    return trajectory


def shoot_numpy_points(
    control_points: np.ndarray,
    physical_momenta: np.ndarray,
    points: np.ndarray,
    origin: np.ndarray,
    scale: float,
    steps: int,
    chunk_size: int,
    threads: int,
) -> np.ndarray:
    """High-accuracy CPU shooting, shared by surface, volume, and replay QA."""
    torch.set_num_threads(threads)
    device = torch.device("cpu")
    dtype = torch.float64
    control = tensor((control_points - origin) / scale, device, dtype)
    momentum = tensor(physical_momenta / scale, device, dtype)
    with torch.no_grad():
        trajectory = control_trajectory(control, momentum, steps)
        pieces = []
        dt = 1.0 / steps
        for start in range(0, len(points), chunk_size):
            moved = tensor((points[start : start + chunk_size] - origin) / scale, device, dtype)
            for current_control, current_momentum, middle_control, middle_momentum in trajectory:
                first = kernel(moved, current_control) @ current_momentum
                midpoint = moved + dt * first / 2.0
                second = kernel(midpoint, middle_control) @ middle_momentum
                moved = moved + dt * second
            pieces.append(moved.cpu().numpy() * scale + origin)
    result = np.concatenate(pieces)
    if result.shape != points.shape or not np.isfinite(result).all():
        raise FloatingPointError("Shooting produced an invalid point array")
    return result


def read_prepared_template(path: Path) -> dict[str, np.ndarray | float]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared CT55 template not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        needed = {
            "points",
            "faces",
            "basal_faces",
            "basal_ids",
            "object_labels",
            "plane_normal",
            "plane_offset",
        }
        missing = sorted(needed.difference(data.files))
        if missing:
            raise ValueError(f"Prepared CT55 template is missing: {missing}")
        result: dict[str, np.ndarray | float] = {
            "points": np.asarray(data["points"], dtype=np.float64),
            "faces": np.asarray(data["faces"], dtype=np.int64),
            "basal_faces": np.asarray(data["basal_faces"], dtype=np.uint8),
            "basal_ids": np.asarray(data["basal_ids"], dtype=np.int64),
            "object_labels": np.asarray(data["object_labels"], dtype=np.int32),
            "plane_normal": np.asarray(data["plane_normal"], dtype=np.float64),
            "plane_offset": float(np.asarray(data["plane_offset"]).reshape(-1)[0]),
        }
    points = result["points"]
    faces = result["faces"]
    basal_faces = result["basal_faces"]
    basal_ids = result["basal_ids"]
    labels = result["object_labels"]
    normal = result["plane_normal"]
    assert isinstance(points, np.ndarray)
    assert isinstance(faces, np.ndarray)
    assert isinstance(basal_faces, np.ndarray)
    assert isinstance(basal_ids, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(normal, np.ndarray)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"Invalid CT55 surface points: {points.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid CT55 surface faces: {faces.shape}")
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("CT55 faces contain invalid point indices")
    if basal_faces.shape != (len(faces),) or labels.shape != (len(faces),):
        raise ValueError("CT55 face labels have incorrect lengths")
    if basal_ids.ndim != 1 or not len(basal_ids):
        raise ValueError("CT55 has no basal vertex labels")
    if np.any(basal_ids < 0) or np.any(basal_ids >= len(points)):
        raise ValueError("CT55 basal indices are invalid")
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("Invalid CT55 basal-plane normal")
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-12:
        raise ValueError("CT55 basal-plane normal has zero length")
    result["plane_normal"] = normal / normal_length
    result["plane_offset"] = float(result["plane_offset"]) / normal_length
    return result


def read_polydata_with_labels(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Surface not found: {path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    surface = reader.GetOutput()
    if surface.GetNumberOfPoints() == 0 or surface.GetNumberOfPolys() == 0:
        raise ValueError(f"Unreadable or empty surface: {path}")
    points = np.asarray(vtk_to_numpy(surface.GetPoints().GetData()), dtype=np.float64)
    cells = surface.GetPolys()
    offsets = np.asarray(vtk_to_numpy(cells.GetOffsetsArray()), dtype=np.int64)
    connectivity = np.asarray(
        vtk_to_numpy(cells.GetConnectivityArray()), dtype=np.int64
    )
    if (
        len(offsets) != surface.GetNumberOfPolys() + 1
        or not np.all(np.diff(offsets) == 3)
    ):
        raise ValueError(f"Surface is not triangle-only: {path}")
    faces = connectivity.reshape(-1, 3)
    labels: dict[str, np.ndarray] = {}
    cell_data = surface.GetCellData()
    for name in ("BasalFace", "AnatomicalObject"):
        array = cell_data.GetArray(name)
        if array is not None:
            labels[name] = np.asarray(vtk_to_numpy(array)).copy()
    return points, faces, labels


def read_template_with_surface_fallback(
    prepared_path: Path,
    original_surface_path: Path,
    labeled_surface_path: Path,
) -> dict[str, np.ndarray | float]:
    if prepared_path.is_file():
        print(f"Template labels:       {prepared_path}")
        return read_prepared_template(prepared_path)

    original_points, original_faces, _ = read_polydata_with_labels(
        original_surface_path
    )
    _, labeled_faces, labels = read_polydata_with_labels(labeled_surface_path)
    missing = sorted({"BasalFace", "AnatomicalObject"}.difference(labels))
    if missing:
        raise ValueError(
            f"{labeled_surface_path} lacks cell arrays {missing}. Keep the labeled "
            "registered CT55 surface or regenerate registration preparation metadata."
        )
    if not np.array_equal(original_faces, labeled_faces):
        raise ValueError(
            "Original and labeled CT55 surfaces have different triangle ordering; "
            "their labels cannot be transferred safely."
        )
    basal_faces = np.asarray(labels["BasalFace"], dtype=np.uint8)
    object_labels = np.asarray(labels["AnatomicalObject"], dtype=np.int32)
    if basal_faces.shape != (len(original_faces),):
        raise ValueError("CT55 BasalFace array has the wrong length")
    if object_labels.shape != (len(original_faces),):
        raise ValueError("CT55 AnatomicalObject array has the wrong length")
    basal_ids = np.unique(original_faces[basal_faces.astype(bool)])
    if len(basal_ids) < 20:
        raise ValueError("CT55 has too few labeled basal vertices")
    basal_points = original_points[basal_ids]
    center = basal_points.mean(axis=0)
    _, _, vh = np.linalg.svd(basal_points - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    offset = float(np.einsum("i,i->", center, normal, optimize=False))
    maximum_error = float(
        np.max(np.abs(np.einsum("ij,j->i", basal_points, normal, optimize=False) - offset))
    )
    if maximum_error > 0.15:
        raise ValueError(
            f"Transferred CT55 basal labels are not planar on the original surface: "
            f"maximum error={maximum_error:.6g} mm"
        )
    print(
        f"Template labels:       transferred from {labeled_surface_path} "
        f"to {original_surface_path}; plane error={maximum_error:.6g} mm"
    )
    return {
        "points": original_points,
        "faces": original_faces,
        "basal_faces": basal_faces,
        "basal_ids": basal_ids.astype(np.int64),
        "object_labels": object_labels,
        "plane_normal": normal,
        "plane_offset": offset,
    }


def read_volume(path: Path) -> vtk.vtkUnstructuredGrid:
    if not path.is_file():
        raise FileNotFoundError(f"Aligned CT55 volume not found: {path}")
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = vtk.vtkUnstructuredGrid()
    grid.DeepCopy(reader.GetOutput())
    if grid.GetNumberOfPoints() == 0 or grid.GetNumberOfCells() == 0:
        raise ValueError(f"Unreadable or empty VTU: {path}")
    points = vtk_to_numpy(grid.GetPoints().GetData())
    if points.shape != (grid.GetNumberOfPoints(), 3) or not np.isfinite(points).all():
        raise ValueError("CT55 volume contains invalid points")
    return grid


def volume_array_signature(grid: vtk.vtkUnstructuredGrid) -> dict[str, list[tuple[str, int]]]:
    result: dict[str, list[tuple[str, int]]] = {}
    for name, container in (
        ("point_data", grid.GetPointData()),
        ("cell_data", grid.GetCellData()),
        ("field_data", grid.GetFieldData()),
    ):
        arrays = []
        for index in range(container.GetNumberOfArrays()):
            array = container.GetAbstractArray(index)
            arrays.append((array.GetName() or f"unnamed_{index}", array.GetNumberOfTuples()))
        result[name] = arrays
    return result


def tetra_connectivity(grid: vtk.vtkUnstructuredGrid) -> np.ndarray:
    cell_array = grid.GetCells()
    offsets = np.asarray(vtk_to_numpy(cell_array.GetOffsetsArray()), dtype=np.int64)
    connectivity = np.asarray(
        vtk_to_numpy(cell_array.GetConnectivityArray()), dtype=np.int64
    )
    cell_types = np.fromiter(
        (grid.GetCellType(index) for index in range(grid.GetNumberOfCells())),
        dtype=np.int16,
        count=grid.GetNumberOfCells(),
    )
    if not np.all(cell_types == vtk.VTK_TETRA):
        unique, counts = np.unique(cell_types, return_counts=True)
        summary = {int(key): int(value) for key, value in zip(unique, counts)}
        raise ValueError(f"CT55 volume is not tetrahedron-only; cell types: {summary}")
    if len(offsets) != grid.GetNumberOfCells() + 1 or not np.all(np.diff(offsets) == 4):
        raise ValueError("CT55 tetrahedral cell connectivity is malformed")
    tetrahedra = connectivity.reshape(-1, 4)
    if tetrahedra.min() < 0 or tetrahedra.max() >= grid.GetNumberOfPoints():
        raise ValueError("CT55 tetrahedra contain invalid point indices")
    return tetrahedra


def signed_tetra_volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    vertices = points[tetrahedra]
    return np.einsum(
        "ij,ij->i",
        vertices[:, 1] - vertices[:, 0],
        np.cross(
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
    ) / 6.0


def tetrahedron_qa(
    before_points: np.ndarray,
    after_points: np.ndarray,
    tetrahedra: np.ndarray,
) -> dict[str, float | int]:
    before = signed_tetra_volumes(before_points, tetrahedra)
    after = signed_tetra_volumes(after_points, tetrahedra)
    tolerance = max(float(np.max(np.abs(before))) * 1e-12, 1e-15)
    valid_before = np.abs(before) > tolerance
    if not np.any(valid_before):
        raise ValueError("Every CT55 template tetrahedron is degenerate")
    ratios = after[valid_before] / before[valid_before]
    inverted = ratios <= 0
    absolute_ratios = np.abs(after[valid_before]) / np.abs(before[valid_before])
    return {
        "tetrahedra": int(len(tetrahedra)),
        "template_degenerate_tetrahedra": int(np.count_nonzero(~valid_before)),
        "inverted_tetrahedra": int(np.count_nonzero(inverted)),
        "degenerate_tetrahedra": int(
            np.count_nonzero(np.abs(after[valid_before]) <= tolerance)
        ),
        "minimum_signed_volume_ratio": float(np.min(ratios)),
        "p01_absolute_volume_ratio": float(np.percentile(absolute_ratios, 1)),
        "median_absolute_volume_ratio": float(np.median(absolute_ratios)),
        "maximum_absolute_volume_ratio": float(np.max(absolute_ratios)),
    }


def triangle_qa(
    before_points: np.ndarray, after_points: np.ndarray, faces: np.ndarray
) -> dict[str, float | int]:
    before = before_points[faces]
    after = after_points[faces]
    before_area = 0.5 * np.linalg.norm(
        np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0]), axis=1
    )
    after_area = 0.5 * np.linalg.norm(
        np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0]), axis=1
    )
    if np.any(before_area <= 1e-14):
        raise ValueError("CT55 template surface contains degenerate triangles")
    ratios = after_area / before_area
    edges = np.sort(
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "triangles": int(len(faces)),
        "degenerate_triangles": int(np.count_nonzero(after_area <= 1e-14)),
        "minimum_triangle_area_ratio": float(np.min(ratios)),
        "p01_triangle_area_ratio": float(np.percentile(ratios, 1)),
        "maximum_triangle_area_ratio": float(np.max(ratios)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
    }


def make_surface(points: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    surface = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_to_vtk(np.ascontiguousarray(points), deep=True, array_type=vtk.VTK_DOUBLE)
    )
    surface.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(
            np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True
        ),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces).ravel(), deep=True),
    )
    surface.SetPolys(cells)
    return surface


def write_surface(
    path: Path,
    points: np.ndarray,
    faces: np.ndarray,
    basal_faces: np.ndarray,
    basal_ids: np.ndarray,
    object_labels: np.ndarray,
) -> None:
    surface = make_surface(points, faces)
    for name, values, vtk_type in (
        ("BasalFace", basal_faces.astype(np.uint8), vtk.VTK_UNSIGNED_CHAR),
        ("AnatomicalObject", object_labels.astype(np.int32), vtk.VTK_INT),
    ):
        array = numpy_to_vtk(values, deep=True, array_type=vtk_type)
        array.SetName(name)
        surface.GetCellData().AddArray(array)
    basal_vertex = np.zeros(len(points), dtype=np.uint8)
    basal_vertex[basal_ids] = 1
    array = numpy_to_vtk(basal_vertex, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    array.SetName("BasalVertex")
    surface.GetPointData().AddArray(array)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise OSError(f"Could not write surface: {path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    saved = reader.GetOutput()
    if saved.GetNumberOfPoints() != len(points) or saved.GetNumberOfPolys() != len(faces):
        raise RuntimeError(f"Saved surface topology changed: {path}")
    saved_points = np.asarray(vtk_to_numpy(saved.GetPoints().GetData()), dtype=np.float64)
    if not np.array_equal(saved_points, points):
        raise RuntimeError(f"Saved surface coordinates changed: {path}")


def write_volume(
    path: Path,
    template: vtk.vtkUnstructuredGrid,
    points: np.ndarray,
) -> None:
    output = vtk.vtkUnstructuredGrid()
    output.DeepCopy(template)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_to_vtk(np.ascontiguousarray(points), deep=True, array_type=vtk.VTK_DOUBLE)
    )
    output.SetPoints(vtk_points)
    expected_signature = volume_array_signature(template)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(output)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOn()
    if writer.Write() != 1:
        raise OSError(f"Could not write volume: {path}")
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    saved = reader.GetOutput()
    if (
        saved.GetNumberOfPoints() != template.GetNumberOfPoints()
        or saved.GetNumberOfCells() != template.GetNumberOfCells()
    ):
        raise RuntimeError(f"Saved VTU topology changed: {path}")
    if volume_array_signature(saved) != expected_signature:
        raise RuntimeError(f"Saved VTU data arrays changed: {path}")
    saved_points = np.asarray(vtk_to_numpy(saved.GetPoints().GetData()), dtype=np.float64)
    if not np.array_equal(saved_points, points):
        raise RuntimeError(f"Saved VTU coordinates changed: {path}")


def surface_to_volume_mapping(
    surface_points: np.ndarray,
    grid: vtk.vtkUnstructuredGrid,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, float]:
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(grid)
    locator.BuildLocator()
    volume_points = np.asarray(vtk_to_numpy(grid.GetPoints().GetData()), dtype=np.float64)
    indices = np.fromiter(
        (locator.FindClosestPoint(point) for point in surface_points),
        dtype=np.int64,
        count=len(surface_points),
    )
    errors = np.linalg.norm(volume_points[indices] - surface_points, axis=1)
    maximum_error = float(np.max(errors))
    if maximum_error > tolerance:
        raise ValueError(
            f"Prepared CT55 surface is not the boundary of the supplied CT55 VTU; "
            f"maximum nearest-point error={maximum_error:.6g}"
        )
    return indices, maximum_error


def load_momentum(path: Path, template: dict[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "control_points",
            "momenta",
            "origin",
            "scale",
            "steps",
            "plane_normal",
            "plane_offset",
            "template",
            "target",
            "latent",
            "temperature",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        result = {
            "control_points": np.asarray(data["control_points"], dtype=np.float64),
            "momenta": np.asarray(data["momenta"], dtype=np.float64),
            "origin": np.asarray(data["origin"], dtype=np.float64),
            "scale": float(np.asarray(data["scale"]).reshape(-1)[0]),
            "steps": int(np.asarray(data["steps"]).reshape(-1)[0]),
            "plane_normal": np.asarray(data["plane_normal"], dtype=np.float64),
            "plane_offset": float(np.asarray(data["plane_offset"]).reshape(-1)[0]),
            "template": str(np.asarray(data["template"]).reshape(-1)[0]),
            "target": str(np.asarray(data["target"]).reshape(-1)[0]),
            "latent": np.asarray(data["latent"], dtype=np.float64),
            "temperature": float(np.asarray(data["temperature"]).reshape(-1)[0]),
        }
    if result["control_points"].ndim != 2 or result["control_points"].shape[1] != 3:
        raise ValueError(f"Invalid control points in {path}")
    if result["momenta"].shape != result["control_points"].shape:
        raise ValueError(f"Momentum/control shape mismatch in {path}")
    if result["origin"].shape != (3,) or result["plane_normal"].shape != (3,):
        raise ValueError(f"Invalid origin or plane in {path}")
    for name in ("control_points", "momenta", "origin", "plane_normal", "latent"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Non-finite {name} in {path}")
    if result["scale"] <= 0 or result["steps"] < 2:
        raise ValueError(f"Invalid shooting metadata in {path}")
    if result["template"] != "ct_case_0055":
        raise ValueError(f"Unexpected template in {path}: {result['template']}")
    template_normal = np.asarray(template["plane_normal"])
    template_offset = float(template["plane_offset"])
    normal_length = float(np.linalg.norm(result["plane_normal"]))
    if normal_length <= 1e-12:
        raise ValueError(f"Sample plane has zero normal: {path}")
    sample_normal = result["plane_normal"] / normal_length
    sample_offset = result["plane_offset"] / normal_length
    if not np.allclose(sample_normal, template_normal, rtol=0, atol=1e-10):
        raise ValueError(f"Sample plane normal differs from CT55: {path}")
    if not math.isclose(sample_offset, template_offset, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Sample plane height differs from CT55: {path}")
    result["plane_normal"] = sample_normal
    result["plane_offset"] = sample_offset
    return result


def plane_metrics(
    points: np.ndarray,
    basal_ids: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> dict[str, float]:
    residual = np.einsum("ij,j->i", points[basal_ids], normal) - offset
    return {
        "maximum_absolute_error": float(np.max(np.abs(residual))),
        "rms_error": float(np.sqrt(np.mean(residual**2))),
        "mean_signed_error": float(np.mean(residual)),
    }


def select_target_plane(
    raw_points: np.ndarray,
    basal_ids: np.ndarray,
    normal: np.ndarray,
    ct55_offset: float,
    policy: str,
    maximum_shift: float,
) -> dict[str, Any]:
    """Select a flat plane while keeping the CT55 normal fixed.

    For a fixed unit normal, the mean normal coordinate is the least-squares
    plane offset for the raw basal vertices. Clamping prevents a decoded
    outlier from moving the cutting level implausibly far from CT55.
    """
    normal_coordinates = np.einsum(
        "ij,j->i", raw_points[basal_ids], normal, optimize=False
    )
    fitted_offset = float(np.mean(normal_coordinates))
    median_offset = float(np.median(normal_coordinates))
    raw_fitted_shift = fitted_offset - float(ct55_offset)
    if policy == "ct55-fixed":
        selected_offset = float(ct55_offset)
        was_clamped = False
    else:
        selected_shift = float(
            np.clip(raw_fitted_shift, -maximum_shift, maximum_shift)
        )
        selected_offset = float(ct55_offset) + selected_shift
        was_clamped = not math.isclose(
            selected_shift, raw_fitted_shift, rel_tol=0.0, abs_tol=1e-12
        )
    return {
        "policy": policy,
        "normal": np.asarray(normal, dtype=np.float64).tolist(),
        "ct55_offset": float(ct55_offset),
        "raw_least_squares_offset": fitted_offset,
        "raw_median_offset": median_offset,
        "selected_offset": selected_offset,
        "selected_shift_from_ct55": selected_offset - float(ct55_offset),
        "raw_fitted_shift_from_ct55": raw_fitted_shift,
        "maximum_allowed_shift": float(maximum_shift),
        "was_clamped": was_clamped,
    }


def refine_momentum(
    sample: dict[str, Any],
    template: dict[str, Any],
    preservation_ids: np.ndarray,
    target_plane_offset: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    dtype = tensor_dtype(device)
    scale = float(sample["scale"])
    origin = np.asarray(sample["origin"])
    control = tensor((sample["control_points"] - origin) / scale, device, dtype)
    initial = tensor(sample["momenta"] / scale, device, dtype)
    momentum = torch.nn.Parameter(initial.detach().clone())
    points = np.asarray(template["points"])
    basal_ids = np.asarray(template["basal_ids"])
    shape_points = np.vstack((points[preservation_ids], points[basal_ids]))
    shape = tensor((shape_points - origin) / scale, device, dtype)
    preserve_stop = len(preservation_ids)
    normal = tensor(np.asarray(template["plane_normal"]), device, dtype)
    normalized_offset = (
        float(target_plane_offset)
        - float(np.dot(origin, np.asarray(template["plane_normal"])))
    ) / scale

    with torch.no_grad():
        raw_coarse = shoot(control, initial, shape, args.optimization_steps)
        raw_validation = shoot(control, initial, shape, int(sample["steps"]))

    multiplier = torch.zeros(len(basal_ids), device=device, dtype=dtype)
    penalty = float(args.initial_penalty)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    lowest_error: dict[str, Any] | None = None
    target_optimization_error = args.flat_tolerance * args.optimization_flat_fraction

    def evaluate(
        value: torch.Tensor,
        steps: int,
        raw_endpoint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        moved = shoot(control, value, shape, steps)
        preservation = ((moved[:preserve_stop] - raw_endpoint[:preserve_stop]) ** 2).mean()
        basal_delta = moved[preserve_stop:] - raw_endpoint[preserve_stop:]
        basal_normal_delta = torch.einsum(
            "ij,j->i", basal_delta, normal
        )[:, None] * normal[None, :]
        basal_tangent = ((basal_delta - basal_normal_delta) ** 2).mean()
        proximity = ((value - initial) ** 2).mean()
        constraint = torch.einsum(
            "ij,j->i", moved[preserve_stop:], normal
        ) - normalized_offset
        objective = (
            args.anatomy_weight * preservation
            + args.basal_tangent_weight * basal_tangent
            + args.momentum_proximity_weight * proximity
        )
        return {
            "moved": moved,
            "preservation": preservation,
            "basal_tangent": basal_tangent,
            "proximity": proximity,
            "constraint": constraint,
            "objective": objective,
        }

    def validation_candidate(phase: str, block: int) -> dict[str, Any]:
        nonlocal best, lowest_error
        with torch.no_grad():
            values = evaluate(momentum, int(sample["steps"]), raw_validation)
        plane_error = torch_float(values["constraint"].abs().max()) * scale
        plane_rms = torch_float(torch.sqrt((values["constraint"] ** 2).mean())) * scale
        preservation_rms = math.sqrt(max(torch_float(values["preservation"]), 0.0)) * scale
        basal_tangent_rms = (
            math.sqrt(max(torch_float(values["basal_tangent"]), 0.0)) * scale
        )
        candidate = {
            "momentum": momentum.detach().clone(),
            "phase": phase,
            "block": block,
            "maximum_plane_error": plane_error,
            "rms_plane_error": plane_rms,
            "preservation_rms": preservation_rms,
            "basal_tangent_rms": basal_tangent_rms,
            "proximity_mse": torch_float(values["proximity"]),
            "objective": torch_float(values["objective"]),
        }
        if lowest_error is None or plane_error < lowest_error["maximum_plane_error"]:
            lowest_error = candidate
        if plane_error <= args.flat_tolerance and (
            best is None or candidate["objective"] < best["objective"]
        ):
            best = candidate
        return candidate

    def optimize_block(
        phase: str,
        block: int,
        steps: int,
        raw_endpoint: torch.Tensor,
        iterations: int,
    ) -> dict[str, Any]:
        optimizer = torch.optim.LBFGS(
            [momentum],
            lr=args.learning_rate,
            max_iter=iterations,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-8 if dtype == torch.float64 else 1e-6,
            tolerance_change=1e-10 if dtype == torch.float64 else 1e-7,
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            values = evaluate(momentum, steps, raw_endpoint)
            constraint = values["constraint"]
            loss = (
                values["objective"]
                + (multiplier * constraint).mean()
                + 0.5 * penalty * (constraint**2).mean()
            )
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise FloatingPointError("Non-finite flat-base refinement objective")
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            coarse_values = evaluate(momentum, steps, raw_endpoint)
            multiplier.add_(penalty * coarse_values["constraint"])
            optimization_error = (
                torch_float(coarse_values["constraint"].abs().max()) * scale
            )
        candidate = validation_candidate(phase, block)
        row = {
            "phase": phase,
            "block": block,
            "steps": steps,
            "penalty": penalty,
            "optimization_maximum_plane_error": optimization_error,
            "validation_maximum_plane_error": candidate["maximum_plane_error"],
            "validation_rms_plane_error": candidate["rms_plane_error"],
            "anatomy_change_rms": candidate["preservation_rms"],
            "basal_tangent_change_rms": candidate["basal_tangent_rms"],
            "momentum_proximity_mse": candidate["proximity_mse"],
        }
        history.append(row)
        print(row, flush=True)
        return row

    for block in range(1, args.outer_iterations + 1):
        row = optimize_block(
            "coarse_constraint",
            block,
            args.optimization_steps,
            raw_coarse,
            args.inner_iterations,
        )
        if (
            row["validation_maximum_plane_error"] <= target_optimization_error
            and row["anatomy_change_rms"] <= args.maximum_refinement_rms
        ):
            break
        penalty = min(penalty * args.penalty_growth, args.maximum_penalty)

    if best is None or best["maximum_plane_error"] > target_optimization_error:
        penalty = max(penalty, args.maximum_penalty)
        for block in range(1, args.polish_blocks + 1):
            row = optimize_block(
                "validation_step_polish",
                block,
                int(sample["steps"]),
                raw_validation,
                args.polish_iterations,
            )
            if (
                row["validation_maximum_plane_error"] <= target_optimization_error
                and row["anatomy_change_rms"] <= args.maximum_refinement_rms
            ):
                break

    selected = best or lowest_error
    if selected is None:
        raise RuntimeError("Flat-base refinement produced no finite candidate")
    refined = selected["momentum"].detach().cpu().numpy() * scale
    if refined.shape != np.asarray(sample["momenta"]).shape or not np.isfinite(refined).all():
        raise FloatingPointError("Refined momentum is invalid")
    selected_report = {key: value for key, value in selected.items() if key != "momentum"}
    return refined, history, selected_report


def _legacy_refinement_main_disabled() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    dtype = tensor_dtype(device)

    template = read_template_with_surface_fallback(
        args.template_prepared,
        args.template_surface,
        args.template_labeled_surface,
    )
    volume = read_volume(args.template_volume)
    volume_points = np.asarray(vtk_to_numpy(volume.GetPoints().GetData()), dtype=np.float64)
    tetrahedra = tetra_connectivity(volume)
    surface_points = np.asarray(template["points"])
    faces = np.asarray(template["faces"])
    basal_faces = np.asarray(template["basal_faces"])
    basal_ids = np.asarray(template["basal_ids"])
    object_labels = np.asarray(template["object_labels"])
    surface_volume_ids, boundary_mapping_error = surface_to_volume_mapping(
        surface_points, volume
    )

    nonbasal_ids = np.setdiff1d(
        np.arange(len(surface_points), dtype=np.int64), basal_ids, assume_unique=False
    )
    rng = np.random.default_rng(args.seed)
    preservation_count = min(args.preservation_points, len(nonbasal_ids))
    preservation_ids = np.sort(
        rng.choice(nonbasal_ids, preservation_count, replace=False)
    )

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Use another directory or pass --resume after an interrupted run."
        )
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    samples, sampling_diagnostics = decode_and_save_momenta(
        args, template, args.output_dir
    )
    surfaces_dir = args.output_dir / "GeneratedSurfaces"
    volumes_dir = args.output_dir / "GeneratedVolumes"
    reports_dir = args.output_dir / "Reports"
    momenta_dir = args.output_dir / "RefinedMomentas"
    for directory in (surfaces_dir, volumes_dir, reports_dir, momenta_dir):
        directory.mkdir(exist_ok=args.resume)

    print(f"Samples:              {len(samples)}")
    print(f"Latent temperature:   {args.temperature:g}")
    if args.sampling_policy == "aggregate-posterior":
        print("Latent sampling:      learned 61-component aggregate posterior")
    else:
        print("Latent sampling:      antithetic Gaussian pairs with radius guard")
    print(f"Optimization device:  {device} ({dtype})")
    print(f"CT55 surface:         {len(surface_points):,} points; {len(faces):,} triangles")
    print(f"CT55 volume:          {len(volume_points):,} points; {len(tetrahedra):,} tetrahedra")
    print(f"Basal vertices:       {len(basal_ids):,}")
    if args.plane_policy == "sample-offset":
        print("Plane policy:         fixed CT55 normal; bounded sample-specific offset")
        print(
            f"Maximum offset shift: {args.maximum_plane_offset_shift:g} mm from CT55"
        )
    else:
        print("Plane policy:         fixed CT55 normal and offset")
    print(f"Flatness tolerance:   {args.flat_tolerance:g} mm")
    print("No cutting or scaling will be performed.\n")

    rows: list[dict[str, Any]] = []
    reference_control: np.ndarray | None = None
    for sequence, (sample_index, sample) in enumerate(samples, start=1):
        sample_id = f"generated_{sample_index:04d}"
        print(f"[{sequence}/{len(samples)}] {sample_id}", flush=True)
        if reference_control is None:
            reference_control = np.asarray(sample["control_points"])
        elif not np.array_equal(reference_control, sample["control_points"]):
            raise ValueError(f"Control-point order changed for {sample_id}")

        report_path = reports_dir / f"{sample_id}.json"
        refined_path = momenta_dir / f"{sample_id}_refined_momenta.npz"
        surface_path = surfaces_dir / f"{sample_id}.vtk"
        volume_path = volumes_dir / f"{sample_id}.vtu"
        expected_outputs = (refined_path, surface_path, volume_path)
        if report_path.exists():
            if not args.resume:
                raise FileExistsError(f"Existing report: {report_path}")
            prior = json.loads(report_path.read_text(encoding="utf-8"))
            missing_outputs = [str(path) for path in expected_outputs if not path.is_file()]
            if missing_outputs:
                raise FileNotFoundError(
                    f"Completed report exists but outputs are missing: {missing_outputs}"
                )
            prior_surface = prior["surface_quality"]
            prior_volume = prior["volume_quality"]
            prior_plane_selection = prior.get("plane_selection", {})
            rows.append(
                {
                    "sample": sample_id,
                    "status": prior["status"],
                    "raw_plane_max": prior["raw_plane"]["maximum_absolute_error"],
                    "final_plane_max": prior["final_plane"]["maximum_absolute_error"],
                    "target_plane_offset": prior_plane_selection.get(
                        "selected_offset", prior.get("plane_offset")
                    ),
                    "plane_offset_shift": prior_plane_selection.get(
                        "selected_shift_from_ct55", 0.0
                    ),
                    "plane_offset_clamped": prior_plane_selection.get(
                        "was_clamped", False
                    ),
                    "refinement_rms": prior["refinement_surface_rms"],
                    "inverted_tetrahedra": prior_volume["inverted_tetrahedra"],
                    "minimum_tetra_volume_ratio": prior_volume[
                        "minimum_signed_volume_ratio"
                    ],
                    "minimum_triangle_area_ratio": prior_surface[
                        "minimum_triangle_area_ratio"
                    ],
                    "replay_error": prior["replay_maximum_coordinate_error"],
                    "surface": prior["outputs"]["surface_vtk"],
                    "volume": prior["outputs"]["tetrahedral_vtu"],
                }
            )
            print(f"  resumed: preserving completed {prior['status']} result")
            continue
        partial_outputs = [str(path) for path in expected_outputs if path.exists()]
        if partial_outputs:
            raise FileExistsError(
                f"Partial outputs exist without a report: {partial_outputs}"
            )
        raw_surface = shoot_numpy_points(
            sample["control_points"],
            sample["momenta"],
            surface_points,
            sample["origin"],
            sample["scale"],
            sample["steps"],
            args.shooting_chunk_size,
            args.threads,
        )
        plane_selection = select_target_plane(
            raw_surface,
            basal_ids,
            np.asarray(template["plane_normal"]),
            float(template["plane_offset"]),
            args.plane_policy,
            args.maximum_plane_offset_shift,
        )
        target_plane_offset = float(plane_selection["selected_offset"])
        raw_plane = plane_metrics(
            raw_surface,
            basal_ids,
            sample["plane_normal"],
            target_plane_offset,
        )
        raw_plane_against_ct55 = plane_metrics(
            raw_surface,
            basal_ids,
            sample["plane_normal"],
            float(template["plane_offset"]),
        )
        print(
            f"  selected plane shift from CT55: "
            f"{plane_selection['selected_shift_from_ct55']:+.6g} mm"
            f"{' (clamped)' if plane_selection['was_clamped'] else ''}",
            flush=True,
        )
        print(
            f"  raw maximum distance to selected plane: "
            f"{raw_plane['maximum_absolute_error']:.6g} mm",
            flush=True,
        )

        refined_momenta, history, selection = refine_momentum(
            sample,
            template,
            preservation_ids,
            target_plane_offset,
            args,
            device,
        )
        generated_surface = shoot_numpy_points(
            sample["control_points"],
            refined_momenta,
            surface_points,
            sample["origin"],
            sample["scale"],
            sample["steps"],
            args.shooting_chunk_size,
            args.threads,
        )
        generated_volume = shoot_numpy_points(
            sample["control_points"],
            refined_momenta,
            volume_points,
            sample["origin"],
            sample["scale"],
            sample["steps"],
            args.shooting_chunk_size,
            args.threads,
        )
        final_plane = plane_metrics(
            generated_surface,
            basal_ids,
            sample["plane_normal"],
            target_plane_offset,
        )
        surface_quality = triangle_qa(surface_points, generated_surface, faces)
        volume_quality = tetrahedron_qa(volume_points, generated_volume, tetrahedra)
        boundary_consistency = float(
            np.max(
                np.linalg.norm(
                    generated_volume[surface_volume_ids] - generated_surface, axis=1
                )
            )
        )
        boundary_consistency_tolerance = max(
            1e-9, 2.0 * boundary_mapping_error + 1e-10
        )
        refinement_displacement = np.linalg.norm(
            generated_surface - raw_surface, axis=1
        )
        refinement_rms = float(
            np.sqrt(np.mean(np.sum((generated_surface - raw_surface) ** 2, axis=1)))
        )

        np.savez_compressed(
            refined_path,
            control_points=np.asarray(sample["control_points"], dtype=np.float64),
            momenta=refined_momenta.astype(np.float64),
            raw_decoded_momenta=np.asarray(sample["momenta"], dtype=np.float64),
            origin=np.asarray(sample["origin"], dtype=np.float64),
            scale=np.float64(sample["scale"]),
            steps=np.int64(sample["steps"]),
            plane_normal=np.asarray(sample["plane_normal"], dtype=np.float64),
            plane_offset=np.float64(target_plane_offset),
            ct55_plane_offset=np.float64(template["plane_offset"]),
            raw_fitted_plane_offset=np.float64(
                plane_selection["raw_least_squares_offset"]
            ),
            plane_offset_shift_from_ct55=np.float64(
                plane_selection["selected_shift_from_ct55"]
            ),
            plane_offset_was_clamped=np.bool_(plane_selection["was_clamped"]),
            plane_policy=np.asarray(args.plane_policy),
            template="ct_case_0055",
            target=sample_id,
            latent=np.asarray(sample["latent"], dtype=np.float32),
            temperature=np.float32(sample["temperature"]),
            sampling_policy=np.asarray(sample["sampling_policy"]),
            posterior_component_index=np.int64(
                -1
                if sample["posterior_component_index"] is None
                else sample["posterior_component_index"]
            ),
            posterior_component_source=np.asarray(
                ""
                if sample["posterior_component_source"] is None
                else sample["posterior_component_source"]
            ),
            refinement_version=REFINEMENT_VERSION,
            strict_flatness_tolerance=np.float64(args.flat_tolerance),
        )

        write_surface(
            surface_path,
            generated_surface,
            faces,
            basal_faces,
            basal_ids,
            object_labels,
        )
        write_volume(volume_path, volume, generated_volume)

        with np.load(refined_path, allow_pickle=False) as saved:
            replay = shoot_numpy_points(
                np.asarray(saved["control_points"], dtype=np.float64),
                np.asarray(saved["momenta"], dtype=np.float64),
                surface_points,
                np.asarray(saved["origin"], dtype=np.float64),
                float(np.asarray(saved["scale"]).reshape(-1)[0]),
                int(np.asarray(saved["steps"]).reshape(-1)[0]),
                args.shooting_chunk_size,
                args.threads,
            )
        replay_error = float(np.max(np.abs(replay - generated_surface)))

        checks = {
            "flat_base": final_plane["maximum_absolute_error"] <= args.flat_tolerance,
            "plane_offset_bound": abs(
                float(plane_selection["selected_shift_from_ct55"])
            )
            <= args.maximum_plane_offset_shift + 1e-12,
            "anatomy_refinement": refinement_rms <= args.maximum_refinement_rms,
            "surface_nondegenerate": surface_quality["degenerate_triangles"] == 0,
            "surface_area_ratio": surface_quality["minimum_triangle_area_ratio"]
            >= args.minimum_triangle_area_ratio,
            "tetrahedra_nondegenerate": volume_quality["degenerate_tetrahedra"] == 0,
            "tetrahedra_not_inverted": volume_quality["inverted_tetrahedra"] == 0,
            "tetra_volume_ratio": volume_quality["minimum_signed_volume_ratio"]
            >= args.minimum_tetra_volume_ratio,
            "surface_volume_boundary_consistency": boundary_consistency
            <= boundary_consistency_tolerance,
            "momentum_replay": replay_error <= 1e-10,
        }
        passed = all(checks.values())
        report = {
            "status": "ACCEPTED" if passed else "REJECTED",
            "sample": sample_id,
            "generation_checkpoint": str(args.checkpoint),
            "generation_checkpoint_sha256": sample["checkpoint_sha256"],
            "refinement_version": REFINEMENT_VERSION,
            "template": "ct_case_0055",
            "plane_policy": args.plane_policy,
            "plane_selection": plane_selection,
            "cut_performed": False,
            "scale_performed": False,
            "raw_plane": raw_plane,
            "raw_plane_against_ct55": raw_plane_against_ct55,
            "final_plane": final_plane,
            "flat_tolerance": args.flat_tolerance,
            "selection": selection,
            "refinement_surface_rms": refinement_rms,
            "refinement_surface_maximum": float(np.max(refinement_displacement)),
            "surface_quality": surface_quality,
            "volume_quality": volume_quality,
            "surface_volume_boundary_maximum_error": boundary_consistency,
            "surface_volume_boundary_tolerance": boundary_consistency_tolerance,
            "surface_volume_template_mapping_maximum_error": boundary_mapping_error,
            "replay_maximum_coordinate_error": replay_error,
            "anatomical_object_labels": {
                str(int(label)): int(np.count_nonzero(object_labels == label))
                for label in np.unique(object_labels)
            },
            "volume_arrays_preserved": volume_array_signature(volume),
            "checks": checks,
            "optimization_history": history,
            "outputs": {
                "refined_momenta": str(refined_path),
                "surface_vtk": str(surface_path),
                "tetrahedral_vtu": str(volume_path),
            },
        }
        write_json(report_path, report)
        row = {
            "sample": sample_id,
            "status": report["status"],
            "raw_plane_max": raw_plane["maximum_absolute_error"],
            "final_plane_max": final_plane["maximum_absolute_error"],
            "target_plane_offset": target_plane_offset,
            "plane_offset_shift": plane_selection["selected_shift_from_ct55"],
            "plane_offset_clamped": plane_selection["was_clamped"],
            "refinement_rms": refinement_rms,
            "inverted_tetrahedra": volume_quality["inverted_tetrahedra"],
            "minimum_tetra_volume_ratio": volume_quality[
                "minimum_signed_volume_ratio"
            ],
            "minimum_triangle_area_ratio": surface_quality[
                "minimum_triangle_area_ratio"
            ],
            "replay_error": replay_error,
            "surface": str(surface_path),
            "volume": str(volume_path),
        }
        rows.append(row)
        print(
            f"  {report['status']}: plane={final_plane['maximum_absolute_error']:.6g} mm; "
            f"offset shift={plane_selection['selected_shift_from_ct55']:+.6g} mm; "
            f"refinement RMS={refinement_rms:.6g} mm; "
            f"inverted tets={volume_quality['inverted_tetrahedra']}",
            flush=True,
        )

    with (args.output_dir / "generation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "generation_complete",
        "processed": len(rows),
        "accepted": sum(row["status"] == "ACCEPTED" for row in rows),
        "rejected": sum(row["status"] == "REJECTED" for row in rows),
        "plane_policy": args.plane_policy,
        "maximum_plane_offset_shift": args.maximum_plane_offset_shift,
        "cut_performed": False,
        "template_source": (
            str(args.template_prepared)
            if args.template_prepared.is_file()
            else str(args.template_surface)
        ),
        "template_source_sha256": (
            sha256(args.template_prepared)
            if args.template_prepared.is_file()
            else sha256(args.template_surface)
        ),
        "template_label_source": (
            str(args.template_prepared)
            if args.template_prepared.is_file()
            else str(args.template_labeled_surface)
        ),
        "template_label_source_sha256": (
            sha256(args.template_prepared)
            if args.template_prepared.is_file()
            else sha256(args.template_labeled_surface)
        ),
        "template_volume": str(args.template_volume),
        "template_volume_sha256": sha256(args.template_volume),
        "flat_tolerance": args.flat_tolerance,
        "sampling": sampling_diagnostics,
        "rows": rows,
    }
    write_json(args.output_dir / "generation_summary.json", summary)
    print(
        f"\nGeneration complete: {summary['accepted']}/{summary['processed']} accepted. "
        f"Review {args.output_dir / 'generation_summary.csv'} and all QA reports."
    )
    return 0 if summary["rejected"] == 0 else 2


def deformed_grid(
    template_grid: vtk.vtkUnstructuredGrid, points: np.ndarray
) -> vtk.vtkUnstructuredGrid:
    grid = vtk.vtkUnstructuredGrid()
    grid.DeepCopy(template_grid)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_to_vtk(np.ascontiguousarray(points), deep=True, array_type=vtk.VTK_DOUBLE)
    )
    grid.SetPoints(vtk_points)
    return grid


def coordinates_along_normal(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return n.x without dispatching the small product through BLAS.

    Some macOS Accelerate/NumPy combinations emit spurious overflow warnings
    for ``points @ normal`` even when both arrays are finite.  ``einsum`` is
    deterministic here and avoids that code path.
    """
    points = np.asarray(points, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normal.shape != (3,):
        raise ValueError("Plane-coordinate inputs must have shapes (N, 3) and (3,)")
    if not np.isfinite(points).all() or not np.isfinite(normal).all():
        raise FloatingPointError("Plane-coordinate inputs contain non-finite values")
    return np.einsum("ij,j->i", points, normal, optimize=False)


def choose_complete_cut_plane(
    raw_surface: np.ndarray,
    basal_ids: np.ndarray,
    normal: np.ndarray,
    ct55_offset: float,
    maximum_shift: float,
    clearance: float,
) -> dict[str, Any]:
    """Put the plane just inside the deepest basal vertex.

    The retained side is inferred from the non-basal surface.  Moving past the
    extreme basal coordinate (rather than its mean) guarantees that none of the
    old, non-planar basal cap remains in the retained mesh.
    """
    all_ids = np.arange(len(raw_surface), dtype=np.int64)
    nonbasal_ids = np.setdiff1d(all_ids, basal_ids, assume_unique=False)
    basal_coordinate = coordinates_along_normal(raw_surface[basal_ids], normal)
    nonbasal_coordinate = coordinates_along_normal(raw_surface[nonbasal_ids], normal)
    side_measure = float(np.median(nonbasal_coordinate) - np.median(basal_coordinate))
    if abs(side_measure) <= 1e-10:
        raise ValueError("Cannot determine the inward side of the generated heart")
    retained_sign = 1 if side_measure > 0 else -1
    if retained_sign > 0:
        required_offset = float(np.max(basal_coordinate) + clearance)
    else:
        required_offset = float(np.min(basal_coordinate) - clearance)
    shift = required_offset - float(ct55_offset)
    within_bound = abs(shift) <= maximum_shift + 1e-12
    selected_shift = float(np.clip(shift, -maximum_shift, maximum_shift))
    selected_offset = float(ct55_offset) + selected_shift
    clearance_values = retained_sign * (selected_offset - basal_coordinate)
    return {
        "policy": "quality-aware-complete-inward-volume-cut",
        "normal": np.asarray(normal, dtype=np.float64).tolist(),
        "ct55_offset": float(ct55_offset),
        "selected_offset": selected_offset,
        "selected_shift_from_ct55": selected_shift,
        "required_unbounded_offset": required_offset,
        "required_unbounded_shift_from_ct55": shift,
        "maximum_allowed_shift": float(maximum_shift),
        "retained_implicit_sign": retained_sign,
        "cut_clearance": float(clearance),
        "minimum_achieved_clearance": float(np.min(clearance_values)),
        "complete_cut_within_bound": bool(within_bound),
        "was_clamped": not within_bound,
    }


def candidate_cut_offsets(
    shot_volume_points: np.ndarray,
    plane: dict[str, Any],
    ct55_offset: float,
    maximum_shift: float,
    search_distance: float,
    search_step: float,
    maximum_candidates: int,
) -> list[dict[str, float]]:
    """Construct shallow-to-deep, vertex-avoiding candidate planes.

    A cut close to a tetrahedral vertex produces intersection points close to
    an edge endpoint and can create arbitrarily poor slivers.  In addition to
    uniformly spaced offsets, this routine inserts the midpoint of the local
    interval between projected volume vertices.  The actual clipped mesh is
    still evaluated; this geometric construction is only a candidate proposal.
    """
    retained_sign = int(plane["retained_implicit_sign"])
    required_offset = float(plane["required_unbounded_offset"])
    directional_bound = float(ct55_offset) + retained_sign * float(maximum_shift)
    available_to_bound = retained_sign * (directional_bound - required_offset)

    if available_to_bound < -1e-12:
        clamped = float(ct55_offset) + float(
            np.clip(required_offset - ct55_offset, -maximum_shift, maximum_shift)
        )
        return [
            {
                "offset": clamped,
                "extra_inward_depth": retained_sign * (clamped - required_offset),
                "nearest_volume_vertex_distance": float(
                    np.min(
                        np.abs(
                            coordinates_along_normal(shot_volume_points, np.asarray(plane["normal"]))
                            - clamped
                        )
                    )
                ),
            }
        ]

    maximum_depth = min(float(search_distance), max(0.0, float(available_to_bound)))
    count = int(math.floor(maximum_depth / search_step + 1e-12))
    depths = [index * search_step for index in range(count + 1)]
    if not depths or maximum_depth - depths[-1] > 1e-12:
        depths.append(maximum_depth)

    normal = np.asarray(plane["normal"], dtype=np.float64)
    projected = np.sort(coordinates_along_normal(shot_volume_points, normal))
    low = min(required_offset, required_offset + retained_sign * maximum_depth)
    high = max(required_offset, required_offset + retained_sign * maximum_depth)
    proposed_offsets = {required_offset + retained_sign * depth for depth in depths}

    # Snap every uniform proposal to the midpoint of its enclosing projected-
    # vertex interval.  This maximizes its local distance from both endpoints.
    for depth in depths:
        target = required_offset + retained_sign * depth
        insertion = int(np.searchsorted(projected, target, side="left"))
        if 0 < insertion < len(projected):
            midpoint = 0.5 * float(projected[insertion - 1] + projected[insertion])
            if low - 1e-12 <= midpoint <= high + 1e-12:
                proposed_offsets.add(midpoint)

    candidates: list[dict[str, float]] = []
    for offset in proposed_offsets:
        extra_depth = retained_sign * (float(offset) - required_offset)
        if extra_depth < -1e-10 or extra_depth > maximum_depth + 1e-10:
            continue
        candidates.append(
            {
                "offset": float(offset),
                "extra_inward_depth": max(0.0, float(extra_depth)),
                "nearest_volume_vertex_distance": float(
                    np.min(np.abs(projected - float(offset)))
                ),
            }
        )
    candidates.sort(
        key=lambda item: (item["extra_inward_depth"], -item["nearest_volume_vertex_distance"])
    )

    # Remove numerically duplicate offsets without changing shallow-first order.
    unique: list[dict[str, float]] = []
    for candidate in candidates:
        if not unique or abs(candidate["offset"] - unique[-1]["offset"]) > 1e-10:
            unique.append(candidate)
    if len(unique) > maximum_candidates:
        indices = np.linspace(0, len(unique) - 1, maximum_candidates, dtype=int)
        unique = [unique[int(index)] for index in np.unique(indices)]
        unique.sort(key=lambda item: item["extra_inward_depth"])
    if not unique:
        raise RuntimeError("No valid cut-plane candidates were constructed")
    return unique


def clip_tetrahedral_volume(
    shot_grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    offset: float,
    retained_sign: int,
) -> vtk.vtkUnstructuredGrid:
    plane = vtk.vtkPlane()
    plane.SetNormal(*(float(value) for value in normal))
    plane.SetOrigin(*(float(value) for value in normal * offset))
    clipper = vtk.vtkClipDataSet()
    clipper.SetInputData(shot_grid)
    clipper.SetClipFunction(plane)
    # Required by vtkClipDataSet when an implicit function is used.  Otherwise
    # VTK may clip with an unrelated active scalar array on the input mesh.
    clipper.GenerateClipScalarsOn()
    clipper.UseValueAsOffsetOff()
    clipper.SetValue(0.0)
    clipper.SetMergeTolerance(0.0)
    clipper.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    clipper.GenerateClippedOutputOff()
    if retained_sign < 0:
        clipper.InsideOutOn()
    else:
        clipper.InsideOutOff()
    clipper.Update()

    # VTK 9.x can still emit wedges for some clipped-tetra configurations.
    # Convert those residual 3-D cells explicitly and verify every output cell.
    tetrahedralize = vtk.vtkDataSetTriangleFilter()
    tetrahedralize.SetInputConnection(clipper.GetOutputPort())
    tetrahedralize.TetrahedraOnlyOn()
    tetrahedralize.Update()
    result = vtk.vtkUnstructuredGrid()
    result.DeepCopy(tetrahedralize.GetOutput())
    if result.GetNumberOfPoints() == 0 or result.GetNumberOfCells() == 0:
        raise RuntimeError("Volume clipping produced an empty mesh")
    tetra_connectivity(result)
    return result


def project_numerical_cap_residual(
    grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    offset: float,
) -> dict[str, float | int]:
    """Project only round-off-level cap residuals exactly onto the cut plane."""
    points = np.asarray(vtk_to_numpy(grid.GetPoints().GetData()), dtype=np.float64)
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    tolerance = max(1e-8 * diagonal, 1e-7)
    signed_residual = coordinates_along_normal(points, normal) - float(offset)
    cap = np.abs(signed_residual) <= tolerance
    if np.count_nonzero(cap) < 3:
        raise RuntimeError("Clipped volume has fewer than three planar cap vertices")
    maximum_before = float(np.max(np.abs(signed_residual[cap])))
    projected = points.copy()
    projected[cap] -= signed_residual[cap, None] * np.asarray(normal)[None, :]
    if not np.isfinite(projected).all():
        raise FloatingPointError("Cap projection produced non-finite coordinates")
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_to_vtk(
            np.ascontiguousarray(projected), deep=True, array_type=vtk.VTK_DOUBLE
        )
    )
    grid.SetPoints(vtk_points)
    maximum_after = float(
        np.max(
            np.abs(
                coordinates_along_normal(projected[cap], np.asarray(normal))
                - float(offset)
            )
        )
    )
    return {
        "projected_vertices": int(np.count_nonzero(cap)),
        "coordinate_tolerance": float(tolerance),
        "maximum_residual_before_projection": maximum_before,
        "maximum_residual_after_projection": maximum_after,
    }


def add_flat_base_labels(
    surface: vtk.vtkPolyData,
    normal: np.ndarray,
    offset: float,
    coordinate_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vtk_to_numpy(surface.GetPoints().GetData()), dtype=np.float64)
    cell_array = surface.GetPolys()
    offsets = np.asarray(vtk_to_numpy(cell_array.GetOffsetsArray()), dtype=np.int64)
    connectivity = np.asarray(
        vtk_to_numpy(cell_array.GetConnectivityArray()), dtype=np.int64
    )
    if len(offsets) != surface.GetNumberOfPolys() + 1 or not np.all(
        np.diff(offsets) == 3
    ):
        raise RuntimeError("Extracted boundary is not triangle-only")
    faces = connectivity.reshape(-1, 3)
    residual = np.abs(coordinates_along_normal(points, normal) - offset)
    basal_vertex = residual <= coordinate_tolerance
    basal_face = np.all(basal_vertex[faces], axis=1).astype(np.uint8)
    if not np.any(basal_face):
        raise RuntimeError("Clipped volume has no identifiable planar basal faces")
    for container, name in (
        (surface.GetCellData(), "BasalFace"),
        (surface.GetPointData(), "BasalVertex"),
    ):
        if container.HasArray(name):
            container.RemoveArray(name)
    cell_label = numpy_to_vtk(basal_face, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    cell_label.SetName("BasalFace")
    surface.GetCellData().AddArray(cell_label)
    point_label = numpy_to_vtk(
        basal_vertex.astype(np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    point_label.SetName("BasalVertex")
    surface.GetPointData().AddArray(point_label)
    return basal_face, basal_vertex


def boundary_from_volume(
    grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    offset: float,
) -> tuple[vtk.vtkPolyData, np.ndarray, np.ndarray]:
    boundary = vtk.vtkDataSetSurfaceFilter()
    boundary.SetInputData(grid)
    boundary.PassThroughCellIdsOn()
    boundary.PassThroughPointIdsOn()
    boundary.Update()
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(boundary.GetOutputPort())
    triangulate.PassLinesOff()
    triangulate.PassVertsOff()
    triangulate.Update()
    surface = vtk.vtkPolyData()
    surface.DeepCopy(triangulate.GetOutput())
    scale = max(np.linalg.norm(np.subtract(surface.GetBounds()[1::2], surface.GetBounds()[::2])), 1.0)
    tolerance = max(1e-8 * scale, 1e-7)
    basal_face, basal_vertex = add_flat_base_labels(
        surface, normal, offset, tolerance
    )
    return surface, basal_face, basal_vertex


def clipped_volume_quality(grid: vtk.vtkUnstructuredGrid) -> dict[str, Any]:
    tetrahedra = tetra_connectivity(grid)
    points = np.asarray(vtk_to_numpy(grid.GetPoints().GetData()), dtype=np.float64)
    absolute_volume = np.abs(signed_tetra_volumes(points, tetrahedra))
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    volume_tolerance = diagonal**3 * 1e-15
    def quality_values(setter_name: str) -> np.ndarray:
        quality_filter = vtk.vtkMeshQuality()
        quality_filter.SetInputData(grid)
        getattr(quality_filter, setter_name)()
        quality_filter.SaveCellQualityOn()
        quality_filter.Update()
        quality_array = quality_filter.GetOutput().GetCellData().GetArray("Quality")
        if quality_array is None:
            raise RuntimeError(
                f"VTK did not produce tetrahedral quality values for {setter_name}"
            )
        return np.asarray(vtk_to_numpy(quality_array), dtype=np.float64)

    quality = quality_values("SetTetQualityMeasureToScaledJacobian")
    minimum_angle = quality_values("SetTetQualityMeasureToMinAngle")
    aspect_ratio = quality_values("SetTetQualityMeasureToAspectRatio")
    finite = np.isfinite(quality)
    finite_angle = np.isfinite(minimum_angle)
    finite_aspect = np.isfinite(aspect_ratio)
    return {
        "points": int(grid.GetNumberOfPoints()),
        "tetrahedra": int(grid.GetNumberOfCells()),
        "nonfinite_scaled_jacobians": int(np.count_nonzero(~finite)),
        "nonpositive_scaled_jacobians": int(np.count_nonzero(quality[finite] <= 0)),
        "minimum_scaled_jacobian": float(np.min(quality[finite])) if np.any(finite) else None,
        "p01_scaled_jacobian": float(np.percentile(quality[finite], 1)) if np.any(finite) else None,
        "median_scaled_jacobian": float(np.median(quality[finite])) if np.any(finite) else None,
        "minimum_dihedral_angle": (
            float(np.min(minimum_angle[finite_angle])) if np.any(finite_angle) else None
        ),
        "p01_dihedral_angle": (
            float(np.percentile(minimum_angle[finite_angle], 1))
            if np.any(finite_angle)
            else None
        ),
        "maximum_aspect_ratio": (
            float(np.max(aspect_ratio[finite_aspect])) if np.any(finite_aspect) else None
        ),
        "degenerate_tetrahedra": int(np.count_nonzero(absolute_volume <= volume_tolerance)),
        "minimum_absolute_volume": float(np.min(absolute_volume)),
        "median_absolute_volume": float(np.median(absolute_volume)),
    }


def extracted_surface_quality(
    surface: vtk.vtkPolyData,
    basal_face: np.ndarray,
    basal_vertex: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> dict[str, Any]:
    points = np.asarray(vtk_to_numpy(surface.GetPoints().GetData()), dtype=np.float64)
    offsets = np.asarray(
        vtk_to_numpy(surface.GetPolys().GetOffsetsArray()), dtype=np.int64
    )
    connectivity = np.asarray(
        vtk_to_numpy(surface.GetPolys().GetConnectivityArray()), dtype=np.int64
    )
    if not np.all(np.diff(offsets) == 3):
        raise RuntimeError("Extracted boundary contains non-triangular polygons")
    faces = connectivity.reshape(-1, 3)
    vertices = points[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=1,
    )
    edges = np.sort(
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    basal_residual = np.abs(
        coordinates_along_normal(points[basal_vertex], normal) - offset
    )
    return {
        "points": int(len(points)),
        "triangles": int(len(faces)),
        "basal_triangles": int(np.count_nonzero(basal_face)),
        "basal_vertices": int(np.count_nonzero(basal_vertex)),
        "maximum_basal_plane_error": float(np.max(basal_residual)),
        "rms_basal_plane_error": float(np.sqrt(np.mean(basal_residual**2))),
        "degenerate_triangles": int(np.count_nonzero(area <= 1e-14)),
        "minimum_triangle_area": float(np.min(area)),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(edge_counts > 2)),
    }


def quality_aware_clip_search(
    shot_grid: vtk.vtkUnstructuredGrid,
    shot_volume_points: np.ndarray,
    normal: np.ndarray,
    plane: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    vtk.vtkUnstructuredGrid,
    vtk.vtkPolyData,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Select the shallowest candidate that passes post-clip mesh QA.

    The test is performed on the actual tetrahedra returned by VTK, not on a
    proxy objective.  If no candidate passes, the least-bad candidate is
    returned but explicitly marked as a failed search so the final sample is
    rejected rather than silently weakening the quality threshold.
    """
    candidates = candidate_cut_offsets(
        shot_volume_points,
        plane,
        float(plane["ct55_offset"]),
        args.maximum_plane_offset_shift,
        args.cut_search_distance,
        args.cut_search_step,
        args.cut_search_maximum_candidates,
    )
    retained_sign = int(plane["retained_implicit_sign"])
    history: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    fallback_score: tuple[float, ...] | None = None

    for index, candidate in enumerate(candidates, start=1):
        offset = float(candidate["offset"])
        extra_depth = float(candidate["extra_inward_depth"])
        minimum_clearance = float(plane["cut_clearance"]) + extra_depth
        try:
            clipped = clip_tetrahedral_volume(
                shot_grid, normal, offset, retained_sign
            )
            projection = project_numerical_cap_residual(clipped, normal, offset)
            volume_quality = clipped_volume_quality(clipped)
            boundary, basal_face, basal_vertex = boundary_from_volume(
                clipped, normal, offset
            )
            surface_quality = extracted_surface_quality(
                boundary, basal_face, basal_vertex, normal, offset
            )
            candidate_checks = {
                "complete_cut": bool(plane["complete_cut_within_bound"])
                and minimum_clearance >= args.cut_clearance - 1e-10,
                "finite_tetrahedral_quality": volume_quality[
                    "nonfinite_scaled_jacobians"
                ]
                == 0,
                "positive_tetrahedral_quality": volume_quality[
                    "nonpositive_scaled_jacobians"
                ]
                == 0,
                "nondegenerate_tetrahedra": volume_quality[
                    "degenerate_tetrahedra"
                ]
                == 0,
                "minimum_scaled_jacobian": volume_quality[
                    "minimum_scaled_jacobian"
                ]
                is not None
                and volume_quality["minimum_scaled_jacobian"]
                >= args.minimum_scaled_jacobian,
                "flat_base": surface_quality["maximum_basal_plane_error"]
                <= args.flat_tolerance,
                "closed_surface": surface_quality["boundary_edges"] == 0,
                "surface_manifold": surface_quality["nonmanifold_edges"] == 0,
                "surface_nondegenerate": surface_quality["degenerate_triangles"]
                == 0,
            }
            passed = all(candidate_checks.values())
            record = {
                "candidate": index,
                "offset": offset,
                "shift_from_ct55": offset - float(plane["ct55_offset"]),
                "extra_inward_depth": extra_depth,
                "minimum_basal_clearance": minimum_clearance,
                "nearest_volume_vertex_distance": candidate[
                    "nearest_volume_vertex_distance"
                ],
                "tetrahedra": volume_quality["tetrahedra"],
                "minimum_scaled_jacobian": volume_quality[
                    "minimum_scaled_jacobian"
                ],
                "p01_scaled_jacobian": volume_quality["p01_scaled_jacobian"],
                "nonfinite_scaled_jacobians": volume_quality[
                    "nonfinite_scaled_jacobians"
                ],
                "nonpositive_scaled_jacobians": volume_quality[
                    "nonpositive_scaled_jacobians"
                ],
                "degenerate_tetrahedra": volume_quality[
                    "degenerate_tetrahedra"
                ],
                "maximum_basal_plane_error": surface_quality[
                    "maximum_basal_plane_error"
                ],
                "checks": candidate_checks,
                "passed": passed,
            }
            history.append(record)
            bundle = {
                "clipped": clipped,
                "boundary": boundary,
                "basal_face": basal_face,
                "basal_vertex": basal_vertex,
                "volume_quality": volume_quality,
                "surface_quality": surface_quality,
                "projection": projection,
                "candidate": candidate,
                "checks": candidate_checks,
                "passed": passed,
            }
            minimum_quality = volume_quality["minimum_scaled_jacobian"]
            percentile_quality = volume_quality["p01_scaled_jacobian"]
            score = (
                float(volume_quality["nonfinite_scaled_jacobians"] == 0),
                float(volume_quality["nonpositive_scaled_jacobians"] == 0),
                float(volume_quality["degenerate_tetrahedra"] == 0),
                float(minimum_quality) if minimum_quality is not None else -1.0,
                float(percentile_quality) if percentile_quality is not None else -1.0,
                -extra_depth,
            )
            if fallback_score is None or score > fallback_score:
                fallback = bundle
                fallback_score = score
            if passed:
                selected = bundle
                break
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            history.append(
                {
                    "candidate": index,
                    "offset": offset,
                    "shift_from_ct55": offset - float(plane["ct55_offset"]),
                    "extra_inward_depth": extra_depth,
                    "minimum_basal_clearance": minimum_clearance,
                    "nearest_volume_vertex_distance": candidate[
                        "nearest_volume_vertex_distance"
                    ],
                    "passed": False,
                    "error": str(exc),
                }
            )

    if selected is None:
        selected = fallback
    if selected is None:
        raise RuntimeError("Every candidate cut failed before producing a QA mesh")

    selected_candidate = selected["candidate"]
    selected_offset = float(selected_candidate["offset"])
    selected_extra = float(selected_candidate["extra_inward_depth"])
    selected_plane = dict(plane)
    selected_plane.update(
        {
            "selected_offset": selected_offset,
            "selected_shift_from_ct55": selected_offset
            - float(plane["ct55_offset"]),
            "quality_search_extra_inward_depth": selected_extra,
            "minimum_achieved_clearance": float(plane["cut_clearance"])
            + selected_extra,
            "was_adjusted_for_tetrahedral_quality": selected_extra > 1e-12,
            "quality_candidate_found": bool(selected["passed"]),
        }
    )
    search_report = {
        "method": "shallowest-passing-actual-mesh-search",
        "search_distance": float(args.cut_search_distance),
        "nominal_step": float(args.cut_search_step),
        "maximum_candidates": int(args.cut_search_maximum_candidates),
        "candidates_available": len(candidates),
        "candidates_tested": len(history),
        "quality_threshold": float(args.minimum_scaled_jacobian),
        "passing_candidate_found": bool(selected["passed"]),
        "selection_basis": (
            "shallowest_passing_candidate"
            if selected["passed"]
            else "best_available_candidate_rejected"
        ),
        "selected_projection": selected["projection"],
        "candidates": history,
    }
    return (
        selected["clipped"],
        selected["boundary"],
        selected["basal_face"],
        selected["basal_vertex"],
        selected["volume_quality"],
        selected["surface_quality"],
        selected_plane,
        search_report,
    )


def write_unstructured_grid(path: Path, grid: vtk.vtkUnstructuredGrid) -> None:
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOn()
    if writer.Write() != 1:
        raise OSError(f"Could not write clipped volume: {path}")
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    saved = reader.GetOutput()
    if (
        saved.GetNumberOfPoints() != grid.GetNumberOfPoints()
        or saved.GetNumberOfCells() != grid.GetNumberOfCells()
    ):
        raise RuntimeError(f"Saved clipped-volume topology changed: {path}")
    tetra_connectivity(saved)
    expected_points = np.asarray(
        vtk_to_numpy(grid.GetPoints().GetData()), dtype=np.float64
    )
    saved_points = np.asarray(
        vtk_to_numpy(saved.GetPoints().GetData()), dtype=np.float64
    )
    if not np.array_equal(saved_points, expected_points):
        raise RuntimeError(f"Saved clipped-volume coordinates changed: {path}")


def write_polydata(path: Path, surface: vtk.vtkPolyData) -> None:
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise OSError(f"Could not write extracted surface: {path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    saved = reader.GetOutput()
    if (
        saved.GetNumberOfPoints() != surface.GetNumberOfPoints()
        or saved.GetNumberOfPolys() != surface.GetNumberOfPolys()
    ):
        raise RuntimeError(f"Saved extracted-surface topology changed: {path}")
    expected_points = np.asarray(
        vtk_to_numpy(surface.GetPoints().GetData()), dtype=np.float64
    )
    saved_points = np.asarray(
        vtk_to_numpy(saved.GetPoints().GetData()), dtype=np.float64
    )
    if not np.array_equal(saved_points, expected_points):
        raise RuntimeError(f"Saved extracted-surface coordinates changed: {path}")


def polydata_points_and_faces(
    surface: vtk.vtkPolyData,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a strict triangle representation of a vtkPolyData."""
    if surface.GetNumberOfPoints() == 0 or surface.GetNumberOfPolys() == 0:
        raise ValueError("Surface is empty")
    points = np.asarray(vtk_to_numpy(surface.GetPoints().GetData()), dtype=np.float64)
    offsets = np.asarray(
        vtk_to_numpy(surface.GetPolys().GetOffsetsArray()), dtype=np.int64
    )
    connectivity = np.asarray(
        vtk_to_numpy(surface.GetPolys().GetConnectivityArray()), dtype=np.int64
    )
    if len(offsets) != surface.GetNumberOfPolys() + 1 or not np.all(
        np.diff(offsets) == 3
    ):
        raise ValueError("Surface is not triangle-only")
    faces = connectivity.reshape(-1, 3)
    if points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Surface contains invalid coordinates")
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("Surface contains invalid triangle indices")
    return points, faces


def basal_cap_topology(
    faces: np.ndarray, basal_face: np.ndarray
) -> dict[str, Any]:
    """Measure the contour topology of the planar myocardial cross-section."""
    cap_faces = np.asarray(faces[np.asarray(basal_face, dtype=bool)], dtype=np.int64)
    if len(cap_faces) == 0:
        return {
            "triangles": 0,
            "boundary_edges": 0,
            "boundary_vertices": 0,
            "contour_loops": 0,
            "every_boundary_vertex_has_degree_two": False,
        }
    edges = np.sort(
        np.vstack(
            (cap_faces[:, [0, 1]], cap_faces[:, [1, 2]], cap_faces[:, [2, 0]])
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    if len(boundary) == 0:
        return {
            "triangles": int(len(cap_faces)),
            "boundary_edges": 0,
            "boundary_vertices": 0,
            "contour_loops": 0,
            "every_boundary_vertex_has_degree_two": False,
        }
    vertices, degree = np.unique(boundary.ravel(), return_counts=True)
    adjacency: dict[int, list[int]] = {int(value): [] for value in vertices}
    for first, second in boundary:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    unseen = set(adjacency)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    degree_two = bool(np.all(degree == 2))
    return {
        "triangles": int(len(cap_faces)),
        "boundary_edges": int(len(boundary)),
        "boundary_vertices": int(len(vertices)),
        "contour_loops": int(components if degree_two else 0),
        "connected_boundary_components": int(components),
        "every_boundary_vertex_has_degree_two": degree_two,
    }


def clip_and_cap_surface(
    raw_points: np.ndarray,
    faces: np.ndarray,
    normal: np.ndarray,
    offset: float,
    retained_sign: int,
    flat_tolerance: float,
) -> tuple[vtk.vtkPolyData, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Clip a closed surface and let VTK triangulate its planar cap."""
    source = make_surface(raw_points, faces)
    plane = vtk.vtkPlane()
    plane.SetNormal(*(float(value) for value in normal))
    plane.SetOrigin(*(float(value) for value in normal * offset))
    planes = vtk.vtkPlaneCollection()
    planes.AddItem(plane)
    clipper = vtk.vtkClipClosedSurface()
    clipper.SetInputData(source)
    clipper.SetClippingPlanes(planes)
    clipper.GenerateFacesOn()
    clipper.GenerateOutlineOff()
    clipper.SetInsideOut(retained_sign < 0)
    diagonal = max(float(np.linalg.norm(np.ptp(raw_points, axis=0))), 1.0)
    geometric_tolerance = max(diagonal * 1e-10, 1e-9)
    clipper.SetTolerance(geometric_tolerance)
    clipper.Update()

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(clipper.GetOutputPort())
    triangulate.PassLinesOff()
    triangulate.PassVertsOff()
    triangulate.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangulate.GetOutputPort())
    clean.PointMergingOn()
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(geometric_tolerance)
    clean.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOff()
    normals.Update()
    surface = vtk.vtkPolyData()
    surface.DeepCopy(normals.GetOutput())

    points, clipped_faces = polydata_points_and_faces(surface)
    residual = coordinates_along_normal(points, normal) - float(offset)
    classification_tolerance = max(flat_tolerance * 10.0, diagonal * 1e-9, 1e-8)
    planar = np.abs(residual) <= classification_tolerance
    if np.count_nonzero(planar) < 3:
        raise RuntimeError("Surface clipping produced no identifiable planar cap")
    points[planar] -= residual[planar, None] * normal[None, :]
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_to_vtk(np.ascontiguousarray(points), deep=True, array_type=vtk.VTK_DOUBLE)
    )
    surface.SetPoints(vtk_points)
    basal_face, basal_vertex = add_flat_base_labels(
        surface, normal, offset, classification_tolerance
    )
    quality = extracted_surface_quality(
        surface, basal_face, basal_vertex, normal, offset
    )
    cap_topology = basal_cap_topology(clipped_faces, basal_face)
    retained_coordinate = retained_sign * (
        coordinates_along_normal(points, normal) - float(offset)
    )
    quality["maximum_retained_side_violation"] = float(
        max(0.0, -float(np.min(retained_coordinate)))
    )
    quality["coordinate_classification_tolerance"] = float(
        classification_tolerance
    )
    return surface, basal_face, basal_vertex, quality, cap_topology


def select_remesh_candidate_records(
    valid_records: list[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    """Mix high-clearance and depth-spanning candidates deterministically."""
    if len(valid_records) <= maximum:
        return sorted(valid_records, key=lambda value: value["extra_inward_depth"])
    selected: dict[int, dict[str, Any]] = {}
    by_depth = sorted(valid_records, key=lambda value: value["extra_inward_depth"])
    selected[int(by_depth[0]["candidate"])] = by_depth[0]
    clearance_count = max(1, maximum // 2)
    by_clearance = sorted(
        valid_records,
        key=lambda value: (
            value["nearest_surface_vertex_distance"],
            -value["extra_inward_depth"],
        ),
        reverse=True,
    )
    for record in by_clearance[:clearance_count]:
        selected[int(record["candidate"])] = record
    for index in np.linspace(0, len(by_depth) - 1, maximum, dtype=int):
        record = by_depth[int(index)]
        selected[int(record["candidate"])] = record
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        for record in by_clearance:
            selected[int(record["candidate"])] = record
            if len(selected) >= maximum:
                break
    return sorted(selected.values(), key=lambda value: value["extra_inward_depth"])


def prescreen_cut_planes(
    raw_surface: np.ndarray,
    template_faces: np.ndarray,
    basal_ids: np.ndarray,
    normal: np.ndarray,
    plane: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = candidate_cut_offsets(
        raw_surface,
        plane,
        float(plane["ct55_offset"]),
        args.maximum_plane_offset_shift,
        args.cut_search_distance,
        args.cut_search_step,
        args.cut_search_maximum_candidates,
    )
    retained_sign = int(plane["retained_implicit_sign"])
    records: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        offset = float(candidate["offset"])
        record: dict[str, Any] = {
            "candidate": index,
            "offset": offset,
            "shift_from_ct55": offset - float(plane["ct55_offset"]),
            "extra_inward_depth": float(candidate["extra_inward_depth"]),
            "nearest_surface_vertex_distance": float(
                candidate["nearest_volume_vertex_distance"]
            ),
        }
        try:
            _, basal_face, _, quality, topology = clip_and_cap_surface(
                raw_surface,
                template_faces,
                normal,
                offset,
                retained_sign,
                args.flat_tolerance,
            )
            minimum_clearance = float(plane["cut_clearance"]) + float(
                candidate["extra_inward_depth"]
            )
            checks = {
                "complete_old_base_removed": bool(plane["complete_cut_within_bound"])
                and minimum_clearance >= args.cut_clearance - 1e-10,
                "flat_cap": quality["maximum_basal_plane_error"]
                <= args.flat_tolerance,
                "closed_surface": quality["boundary_edges"] == 0,
                "manifold_surface": quality["nonmanifold_edges"] == 0,
                "nondegenerate_surface": quality["degenerate_triangles"] == 0,
                "retained_side": quality["maximum_retained_side_violation"]
                <= max(args.flat_tolerance, 1e-8),
                "cap_has_triangles": int(np.count_nonzero(basal_face)) > 0,
                "cap_contours_are_loops": topology[
                    "every_boundary_vertex_has_degree_two"
                ],
                "expected_outer_lv_rv_contours": (
                    args.expected_basal_contours == 0
                    or topology["contour_loops"] == args.expected_basal_contours
                ),
            }
            record.update(
                {
                    "minimum_old_base_clearance": minimum_clearance,
                    "surface_quality": quality,
                    "cap_topology": topology,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
            if record["passed"]:
                valid.append(record)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            record.update({"passed": False, "error": str(exc)})
        records.append(record)
    return records, select_remesh_candidate_records(valid, args.remesh_candidates)


def build_tetrahedral_grid(
    points: np.ndarray, tetrahedra: np.ndarray
) -> vtk.vtkUnstructuredGrid:
    points = np.asarray(points, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Remesher returned invalid points")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or len(tetrahedra) == 0:
        raise ValueError("Remesher returned invalid tetrahedra")
    if tetrahedra.min() < 0 or tetrahedra.max() >= len(points):
        raise ValueError("Remesher returned out-of-range tetrahedron indices")
    signed = signed_tetra_volumes(points, tetrahedra)
    if not np.isfinite(signed).all():
        raise FloatingPointError("Remesher returned non-finite tetrahedra")
    oriented = tetrahedra.copy()
    negative = signed < 0
    oriented[negative, 0], oriented[negative, 1] = (
        tetrahedra[negative, 1],
        tetrahedra[negative, 0],
    )
    signed = signed_tetra_volumes(points, oriented)
    tolerance = max(float(np.linalg.norm(np.ptp(points, axis=0))) ** 3 * 1e-15, 1e-15)
    if np.any(signed <= tolerance):
        raise RuntimeError("Remesher produced zero-volume tetrahedra")
    grid = vtk.vtkUnstructuredGrid()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_to_vtk(np.ascontiguousarray(points), deep=True, array_type=vtk.VTK_DOUBLE)
    )
    grid.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(
            np.arange(0, 4 * len(oriented) + 1, 4, dtype=np.int64), deep=True
        ),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(oriented).ravel(), deep=True),
    )
    grid.SetCells(vtk.VTK_TETRA, cells)
    tetra_connectivity(grid)
    return grid


def _bounded_process_text(value: str | None, maximum: int = 12000) -> str:
    """Keep worker diagnostics useful without making JSON reports enormous."""
    if not value:
        return ""
    text = value.strip()
    if len(text) <= maximum:
        return text
    return text[: maximum // 2] + "\n... output truncated ...\n" + text[-maximum // 2 :]


def _returncode_description(returncode: int) -> str:
    """Translate a subprocess return code into a stable diagnostic string."""
    if returncode < 0:
        number = -returncode
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = f"SIGNAL_{number}"
        return f"terminated by {name} ({number})"
    # Some shells convert a fatal signal to 128 + signal instead of returning
    # the negative signal number exposed by subprocess on POSIX.
    if returncode >= 128:
        number = returncode - 128
        try:
            name = signal.Signals(number).name
            return f"exited {returncode} after {name} ({number})"
        except ValueError:
            pass
    return f"exited with status {returncode}"


def _tetgen_worker_main(argv: list[str]) -> int:
    """Run exactly one TetGen call in a disposable process.

    TetGen is a native extension.  Invalid or numerically difficult PLC input
    can terminate the interpreter with SIGSEGV, which Python exceptions cannot
    catch.  The normal generator therefore invokes this private entry point in
    a child process and trusts output only after a clean zero exit status.
    """
    worker_parser = argparse.ArgumentParser(add_help=False)
    worker_parser.add_argument("request", type=Path)
    worker_parser.add_argument("result", type=Path)
    worker_args = worker_parser.parse_args(argv)
    try:
        import tetgen  # type: ignore

        with np.load(worker_args.request, allow_pickle=False) as request:
            points = np.ascontiguousarray(request["points"], dtype=np.float64)
            faces = np.ascontiguousarray(request["faces"], dtype=np.int32)
            cdt = bool(int(request["cdt"]))
            maximum_volume = float(request["maximum_volume"])
            radius_edge_ratio = float(request["radius_edge_ratio"])
            minimum_dihedral = float(request["minimum_dihedral"])
            optimization_iterations = int(request["optimization_iterations"])
            steiner_points = int(request["steiner_points"])
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("TetGen worker received invalid points")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("TetGen worker received invalid triangles")
        if faces.min() < 0 or faces.max() >= len(points):
            raise ValueError("TetGen worker received out-of-range triangle indices")

        mesher = tetgen.TetGen(points, faces)
        result = mesher.tetrahedralize(
            plc=True,
            quality=True,
            cdt=cdt,
            nobisect=False,
            # TetGen's final consistency checker and optional face-output pass
            # are unnecessary here.  Both traverse the full complex PLC and
            # have caused native crashes for these multiply connected hearts.
            docheck=False,
            quiet=True,
            order=1,
            facesout=False,
            fixedvolume=True,
            maxvolume=maximum_volume,
            minratio=radius_edge_ratio,
            mindihedral=minimum_dihedral,
            opt_iterations=optimization_iterations,
            steinerleft=steiner_points,
        )
        if len(result) < 2:
            raise RuntimeError("TetGen worker returned no tetrahedral connectivity")
        nodes = np.ascontiguousarray(result[0], dtype=np.float64)
        elements = np.ascontiguousarray(result[1], dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
            raise ValueError("TetGen worker returned invalid points")
        if elements.ndim != 2 or elements.shape[1] != 4 or len(elements) == 0:
            raise ValueError("TetGen worker returned invalid tetrahedra")
        if elements.min() < 0 or elements.max() >= len(nodes):
            raise ValueError("TetGen worker returned out-of-range tetrahedron indices")
        np.savez_compressed(
            worker_args.result,
            nodes=nodes,
            elements=elements,
            tetgen_version=np.asarray(str(getattr(tetgen, "__version__", "unknown"))),
        )
        return 0
    except Exception as exc:
        print(f"TetGen worker error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _run_isolated_tetgen_attempt(
    points: np.ndarray,
    faces: np.ndarray,
    maximum_volume: float,
    cdt: bool,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Execute one TetGen mode out of process and validate its serialized result."""
    with tempfile.TemporaryDirectory(prefix="ct55_tetgen_") as temporary:
        temporary_path = Path(temporary)
        request_path = temporary_path / "request.npz"
        result_path = temporary_path / "result.npz"
        np.savez_compressed(
            request_path,
            points=np.ascontiguousarray(points, dtype=np.float64),
            faces=np.ascontiguousarray(faces, dtype=np.int32),
            cdt=np.uint8(cdt),
            maximum_volume=np.float64(maximum_volume),
            radius_edge_ratio=np.float64(args.tetgen_radius_edge_ratio),
            minimum_dihedral=np.float64(args.tetgen_min_dihedral),
            optimization_iterations=np.int64(
                args.tetgen_optimization_iterations
            ),
            steiner_points=np.int64(args.tetgen_steiner_points),
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_tetgen-worker",
            str(request_path),
            str(result_path),
        ]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=temporary_path,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(args.tetgen_timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            stdout = _bounded_process_text(
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout
            )
            stderr = _bounded_process_text(
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr
            )
            raise RuntimeError(
                "isolated TetGen attempt timed out after "
                f"{duration:.1f} s (limit {args.tetgen_timeout_seconds:g} s); "
                f"stdout={stdout!r}; stderr={stderr!r}"
            ) from None
        duration = time.monotonic() - started
        stdout = _bounded_process_text(completed.stdout)
        stderr = _bounded_process_text(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                "isolated TetGen attempt "
                f"{_returncode_description(completed.returncode)} after "
                f"{duration:.1f} s; stdout={stdout!r}; stderr={stderr!r}"
            )
        if not result_path.is_file():
            raise RuntimeError(
                "isolated TetGen worker exited successfully but did not write a result"
            )
        try:
            with np.load(result_path, allow_pickle=False) as result:
                nodes = np.ascontiguousarray(result["nodes"], dtype=np.float64)
                elements = np.ascontiguousarray(result["elements"], dtype=np.int64)
                version = str(np.asarray(result["tetgen_version"]).item())
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"could not validate isolated TetGen result: {exc}") from exc
        return nodes, elements, {
            "backend": "tetgen-python-isolated-subprocess",
            "backend_version": version,
            "wall_time_seconds": float(duration),
            "worker_stdout": stdout,
            "worker_stderr": stderr,
        }


def tetgen_tetrahedralize_surface(
    surface: vtk.vtkPolyData,
    maximum_volume: float,
    args: argparse.Namespace,
) -> tuple[vtk.vtkUnstructuredGrid, dict[str, Any]]:
    points, faces = polydata_points_and_faces(surface)
    attempts = (
        {"name": "plc_delaunay_refinement", "cdt": False},
        {"name": "constrained_delaunay_fallback", "cdt": True},
    )
    failures: list[str] = []
    for attempt in attempts:
        try:
            nodes, elements, worker_report = _run_isolated_tetgen_attempt(
                points,
                faces,
                maximum_volume,
                bool(attempt["cdt"]),
                args,
            )
            grid = build_tetrahedral_grid(nodes, elements)
            return grid, {
                **worker_report,
                "attempt": attempt["name"],
                "input_points": int(len(points)),
                "input_triangles": int(len(faces)),
                "maximum_tetrahedron_volume": float(maximum_volume),
                "radius_edge_ratio_target": float(args.tetgen_radius_edge_ratio),
                "minimum_dihedral_target": float(args.tetgen_min_dihedral),
                "optimization_iterations": int(args.tetgen_optimization_iterations),
                "steiner_point_limit": int(args.tetgen_steiner_points),
                "timeout_seconds": float(args.tetgen_timeout_seconds),
                "previous_attempt_failures": failures,
            }
        except Exception as exc:  # TetGen raises several extension-level errors.
            failures.append(f"{attempt['name']}: {exc}")
    raise RuntimeError("TetGen failed both robust PLC modes: " + " | ".join(failures))


def closest_cell_id(locator: vtk.vtkStaticCellLocator, point: np.ndarray) -> int:
    cell_id = int(locator.FindCell(tuple(float(value) for value in point)))
    if cell_id >= 0:
        return cell_id
    closest = [0.0, 0.0, 0.0]
    cell_reference = vtk.reference(-1)
    sub_id = vtk.reference(0)
    distance2 = vtk.reference(0.0)
    locator.FindClosestPoint(
        tuple(float(value) for value in point),
        closest,
        cell_reference,
        sub_id,
        distance2,
    )
    return int(cell_reference)


def transfer_volume_metadata(
    source: vtk.vtkUnstructuredGrid, target: vtk.vtkUnstructuredGrid
) -> dict[str, Any]:
    """Transfer source point/cell arrays without changing remeshed geometry."""
    copied_point_arrays: list[str] = []
    copied_cell_arrays: list[str] = []
    skipped_arrays: list[str] = []
    source_points = np.asarray(
        vtk_to_numpy(source.GetPoints().GetData()), dtype=np.float64
    )
    target_points = np.asarray(
        vtk_to_numpy(target.GetPoints().GetData()), dtype=np.float64
    )
    point_locator = vtk.vtkStaticPointLocator()
    point_locator.SetDataSet(source)
    point_locator.BuildLocator()
    point_ids = np.fromiter(
        (point_locator.FindClosestPoint(point) for point in target_points),
        dtype=np.int64,
        count=len(target_points),
    )

    source_tetrahedra = tetra_connectivity(source)
    target_tetrahedra = tetra_connectivity(target)
    target_centers = target_points[target_tetrahedra].mean(axis=1)
    cell_locator = vtk.vtkStaticCellLocator()
    cell_locator.SetDataSet(source)
    cell_locator.BuildLocator()
    containing_ids = np.fromiter(
        (cell_locator.FindCell(tuple(float(value) for value in point)) for point in target_centers),
        dtype=np.int64,
        count=len(target_centers),
    )
    outside = containing_ids < 0
    if np.any(outside):
        for index in np.flatnonzero(outside):
            containing_ids[index] = closest_cell_id(cell_locator, target_centers[index])
    if np.any(containing_ids < 0):
        raise RuntimeError("Could not map remeshed tetrahedra to the shot CT55 volume")

    integer_types = {
        vtk.VTK_CHAR,
        vtk.VTK_SIGNED_CHAR,
        vtk.VTK_UNSIGNED_CHAR,
        vtk.VTK_SHORT,
        vtk.VTK_UNSIGNED_SHORT,
        vtk.VTK_INT,
        vtk.VTK_UNSIGNED_INT,
        vtk.VTK_LONG,
        vtk.VTK_UNSIGNED_LONG,
        vtk.VTK_LONG_LONG,
        vtk.VTK_UNSIGNED_LONG_LONG,
        vtk.VTK_ID_TYPE,
        vtk.VTK_BIT,
    }
    for source_data, target_data, indices, destination, point_association in (
        (
            source.GetPointData(),
            target.GetPointData(),
            point_ids,
            copied_point_arrays,
            True,
        ),
        (
            source.GetCellData(),
            target.GetCellData(),
            containing_ids,
            copied_cell_arrays,
            False,
        ),
    ):
        for array_index in range(source_data.GetNumberOfArrays()):
            array = source_data.GetAbstractArray(array_index)
            name = array.GetName() or f"unnamed_{array_index}"
            if not array.IsA("vtkDataArray"):
                skipped_arrays.append(name)
                continue
            # Nearest-neighbour transfer is mathematically appropriate for
            # categorical point labels, but not for old floating-point fields
            # such as normals or a prior simulation solution. Cell fields are
            # sampled from the containing shot tetrahedron and can be copied.
            if point_association and array.GetDataType() not in integer_types:
                skipped_arrays.append(name)
                continue
            try:
                values = np.asarray(vtk_to_numpy(array))[indices]
                if values.dtype == np.bool_:
                    values = values.astype(np.uint8)
                copied = numpy_to_vtk(
                    np.ascontiguousarray(values),
                    deep=True,
                    array_type=array.GetDataType(),
                )
                copied.SetName(name)
                target_data.AddArray(copied)
                destination.append(name)
            except (TypeError, ValueError):
                skipped_arrays.append(name)
    target.GetFieldData().DeepCopy(source.GetFieldData())
    return {
        "method": "nearest-shot-point_and_containing-shot-cell",
        "copied_point_arrays": copied_point_arrays,
        "copied_cell_arrays": copied_cell_arrays,
        "skipped_non_numeric_or_unsupported_arrays": sorted(set(skipped_arrays)),
        "target_cell_centroids_outside_source_before_fallback": int(
            np.count_nonzero(outside)
        ),
        "source_points": int(len(source_points)),
        "source_tetrahedra": int(len(source_tetrahedra)),
    }


def add_anatomical_object_labels(
    final_surface: vtk.vtkPolyData,
    raw_surface_points: np.ndarray,
    raw_surface_faces: np.ndarray,
    raw_object_labels: np.ndarray,
    basal_face: np.ndarray | None = None,
) -> dict[str, Any]:
    source = make_surface(raw_surface_points, raw_surface_faces)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(source)
    locator.BuildLocator()
    points, faces = polydata_points_and_faces(final_surface)
    centers = points[faces].mean(axis=1)
    source_ids = np.empty(len(centers), dtype=np.int64)
    squared_distance = np.empty(len(centers), dtype=np.float64)
    for index, center in enumerate(centers):
        closest = [0.0, 0.0, 0.0]
        cell_reference = vtk.reference(-1)
        sub_id = vtk.reference(0)
        distance2 = vtk.reference(0.0)
        locator.FindClosestPoint(
            tuple(float(value) for value in center),
            closest,
            cell_reference,
            sub_id,
            distance2,
        )
        source_ids[index] = int(cell_reference)
        squared_distance[index] = float(distance2)
    if np.any(source_ids < 0):
        raise RuntimeError("Could not transfer AnatomicalObject surface labels")
    source_labels = np.asarray(raw_object_labels, dtype=np.int32)
    labels = source_labels[source_ids]
    basal_triangles_assigned_zero = 0
    if basal_face is not None:
        basal_face = np.asarray(basal_face, dtype=bool).reshape(-1)
        if basal_face.shape != (len(faces),):
            raise ValueError("Basal-face mask does not match the output surface")
        # Label 0 in the CT55 surface denotes its artificial basal closure,
        # while positive labels identify the outer wall, LV and RV objects.
        # Nearest-wall transfer is meaningful for the anatomical wall only;
        # the newly constructed planar cap must retain the dedicated 0 label.
        if np.any(source_labels == 0):
            labels = labels.copy()
            labels[basal_face] = 0
            basal_triangles_assigned_zero = int(np.count_nonzero(basal_face))
    if final_surface.GetCellData().HasArray("AnatomicalObject"):
        final_surface.GetCellData().RemoveArray("AnatomicalObject")
    array = numpy_to_vtk(labels, deep=True, array_type=vtk.VTK_INT)
    array.SetName("AnatomicalObject")
    final_surface.GetCellData().AddArray(array)
    return {
        "method": "nearest-shot-surface-triangle",
        "source_objects": sorted(int(value) for value in np.unique(raw_object_labels)),
        "output_objects": sorted(int(value) for value in np.unique(labels)),
        "basal_triangles_assigned_zero": basal_triangles_assigned_zero,
        "basal_triangles": (
            int(np.count_nonzero(basal_face)) if basal_face is not None else 0
        ),
        "maximum_triangle_centroid_distance": float(
            np.sqrt(np.max(squared_distance))
        ),
        "rms_triangle_centroid_distance": float(
            np.sqrt(np.mean(squared_distance))
        ),
    }


def bidirectional_surface_fidelity(
    reference: vtk.vtkPolyData, candidate: vtk.vtkPolyData
) -> dict[str, float]:
    def samples(surface: vtk.vtkPolyData) -> np.ndarray:
        points, faces = polydata_points_and_faces(surface)
        return np.vstack((points, points[faces].mean(axis=1)))

    def distances(query: np.ndarray, target: vtk.vtkPolyData) -> np.ndarray:
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(target)
        locator.BuildLocator()
        result = np.empty(len(query), dtype=np.float64)
        for index, point in enumerate(query):
            closest = [0.0, 0.0, 0.0]
            cell_reference = vtk.reference(-1)
            sub_id = vtk.reference(0)
            distance2 = vtk.reference(0.0)
            locator.FindClosestPoint(
                tuple(float(value) for value in point),
                closest,
                cell_reference,
                sub_id,
                distance2,
            )
            result[index] = math.sqrt(max(float(distance2), 0.0))
        return result

    reference_to_candidate = distances(samples(reference), candidate)
    candidate_to_reference = distances(samples(candidate), reference)
    combined = np.concatenate((reference_to_candidate, candidate_to_reference))
    return {
        "maximum_bidirectional_distance": float(np.max(combined)),
        "rms_bidirectional_distance": float(np.sqrt(np.mean(combined**2))),
        "p95_bidirectional_distance": float(np.percentile(combined, 95)),
        "p99_bidirectional_distance": float(np.percentile(combined, 99)),
        "reference_to_candidate_maximum": float(np.max(reference_to_candidate)),
        "candidate_to_reference_maximum": float(np.max(candidate_to_reference)),
    }


def remesh_one_cut_candidate(
    raw_surface: np.ndarray,
    template_faces: np.ndarray,
    template_object_labels: np.ndarray,
    shot_grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    retained_sign: int,
    candidate: dict[str, Any],
    maximum_volume: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    offset = float(candidate["offset"])
    clipped_surface, _, _, clipped_surface_quality, clipped_cap_topology = (
        clip_and_cap_surface(
            raw_surface,
            template_faces,
            normal,
            offset,
            retained_sign,
            args.flat_tolerance,
        )
    )
    grid, tetgen_report = tetgen_tetrahedralize_surface(
        clipped_surface, maximum_volume, args
    )
    projection = project_numerical_cap_residual(grid, normal, offset)
    metadata = transfer_volume_metadata(shot_grid, grid)
    final_surface, basal_face, basal_vertex = boundary_from_volume(
        grid, normal, offset
    )
    anatomical_labels = add_anatomical_object_labels(
        final_surface,
        raw_surface,
        template_faces,
        template_object_labels,
        basal_face,
    )
    volume_quality = clipped_volume_quality(grid)
    surface_quality = extracted_surface_quality(
        final_surface, basal_face, basal_vertex, normal, offset
    )
    final_points, final_faces = polydata_points_and_faces(final_surface)
    final_cap_topology = basal_cap_topology(final_faces, basal_face)
    fidelity = bidirectional_surface_fidelity(clipped_surface, final_surface)
    diagonal = max(float(np.linalg.norm(np.ptp(final_points, axis=0))), 1.0)
    fidelity_tolerance = max(diagonal * 1e-6, 1e-5)
    expected_objects = sorted(
        int(value)
        for value in np.unique(template_object_labels)
        if int(value) > 0
    )
    output_anatomical_objects = sorted(
        int(value)
        for value in anatomical_labels["output_objects"]
        if int(value) > 0
    )
    checks = {
        "positive_tetrahedra": volume_quality["nonpositive_scaled_jacobians"] == 0,
        "finite_tetrahedral_quality": volume_quality[
            "nonfinite_scaled_jacobians"
        ]
        == 0,
        "nondegenerate_tetrahedra": volume_quality["degenerate_tetrahedra"] == 0,
        "minimum_scaled_jacobian": volume_quality["minimum_scaled_jacobian"]
        is not None
        and volume_quality["minimum_scaled_jacobian"]
        >= args.minimum_scaled_jacobian,
        "p01_scaled_jacobian": volume_quality["p01_scaled_jacobian"] is not None
        and volume_quality["p01_scaled_jacobian"]
        >= args.minimum_p01_scaled_jacobian,
        "minimum_dihedral_angle": volume_quality["minimum_dihedral_angle"]
        is not None
        and volume_quality["minimum_dihedral_angle"]
        >= args.minimum_dihedral_angle,
        "flat_base": surface_quality["maximum_basal_plane_error"]
        <= args.flat_tolerance,
        "closed_surface": surface_quality["boundary_edges"] == 0,
        "manifold_surface": surface_quality["nonmanifold_edges"] == 0,
        "nondegenerate_surface": surface_quality["degenerate_triangles"] == 0,
        "expected_outer_lv_rv_contours": (
            args.expected_basal_contours == 0
            or final_cap_topology["contour_loops"]
            == args.expected_basal_contours
        ),
        "surface_matches_clipped_plc": fidelity[
            "maximum_bidirectional_distance"
        ]
        <= fidelity_tolerance,
        "all_anatomical_objects_preserved": output_anatomical_objects
        == expected_objects,
        "basal_cap_has_dedicated_zero_label": (
            0 not in anatomical_labels["source_objects"]
            or anatomical_labels["basal_triangles_assigned_zero"]
            == anatomical_labels["basal_triangles"]
        ),
    }
    return {
        "grid": grid,
        "surface": final_surface,
        "candidate": candidate,
        "tetgen": tetgen_report,
        "cap_projection": projection,
        "volume_metadata": metadata,
        "anatomical_label_transfer": anatomical_labels,
        "clipped_surface_quality": clipped_surface_quality,
        "clipped_cap_topology": clipped_cap_topology,
        "volume_quality": volume_quality,
        "surface_quality": surface_quality,
        "final_cap_topology": final_cap_topology,
        "surface_fidelity": fidelity,
        "surface_fidelity_tolerance": fidelity_tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def remesh_quality_score(bundle: dict[str, Any]) -> tuple[float, ...]:
    quality = bundle["volume_quality"]
    return (
        float(bool(bundle["passed"])),
        float(quality["minimum_scaled_jacobian"] or -1.0),
        float(quality["p01_scaled_jacobian"] or -1.0),
        float(quality["minimum_dihedral_angle"] or -1.0),
        -float(bundle["candidate"]["extra_inward_depth"]),
    )


def quality_driven_surface_chop_and_remesh(
    raw_surface: np.ndarray,
    template: dict[str, Any],
    shot_grid: vtk.vtkUnstructuredGrid,
    shot_volume_points: np.ndarray,
    template_tetrahedra: np.ndarray,
    normal: np.ndarray,
    plane: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    records, selected_records = prescreen_cut_planes(
        raw_surface,
        np.asarray(template["faces"], dtype=np.int64),
        np.asarray(template["basal_ids"], dtype=np.int64),
        normal,
        plane,
        args,
    )
    signed = np.abs(signed_tetra_volumes(shot_volume_points, template_tetrahedra))
    maximum_volume = (
        float(args.tetgen_maximum_volume)
        if args.tetgen_maximum_volume > 0
        else float(np.median(signed)) * float(args.tetgen_volume_factor)
    )
    if not math.isfinite(maximum_volume) or maximum_volume <= 0:
        raise RuntimeError("Automatic TetGen maximum volume is invalid")
    attempts: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    for remesh_index, record in enumerate(selected_records, start=1):
        attempt: dict[str, Any] = {
            "candidate": int(record["candidate"]),
            "offset": float(record["offset"]),
            "shift_from_ct55": float(record["shift_from_ct55"]),
            "extra_inward_depth": float(record["extra_inward_depth"]),
            "nearest_surface_vertex_distance": float(
                record["nearest_surface_vertex_distance"]
            ),
        }
        print(
            f"  remesh candidate {remesh_index}/{len(selected_records)}: "
            f"cut depth={record['extra_inward_depth']:.4g} mm",
            flush=True,
        )
        try:
            bundle = remesh_one_cut_candidate(
                raw_surface,
                np.asarray(template["faces"], dtype=np.int64),
                np.asarray(template["object_labels"], dtype=np.int32),
                shot_grid,
                normal,
                int(plane["retained_implicit_sign"]),
                record,
                maximum_volume,
                args,
            )
            bundles.append(bundle)
            attempt.update(
                {
                    "passed": bool(bundle["passed"]),
                    "checks": bundle["checks"],
                    "tetgen": bundle["tetgen"],
                    "volume_quality": bundle["volume_quality"],
                    "surface_quality": bundle["surface_quality"],
                    "cap_topology": bundle["final_cap_topology"],
                    "surface_fidelity": bundle["surface_fidelity"],
                }
            )
        except Exception as exc:
            attempt.update({"passed": False, "error": str(exc)})
            print(
                "    failed safely: " + _bounded_process_text(str(exc), 1200),
                flush=True,
            )
        attempts.append(attempt)
    best = max(bundles, key=remesh_quality_score) if bundles else None
    search = {
        "method": "surface_clip_planar_cap_then_quality_tetgen_remesh",
        "selection_rule": (
            "best_verified_mesh; minimum scaled Jacobian, p01 scaled Jacobian, "
            "minimum angle, then shallower cut"
        ),
        "candidate_planes_constructed": len(records),
        "surface_valid_candidates": int(sum(bool(record["passed"]) for record in records)),
        "remesh_candidates_requested": int(args.remesh_candidates),
        "remesh_candidates_attempted": len(attempts),
        "passing_remeshes": int(sum(bool(attempt["passed"]) for attempt in attempts)),
        "maximum_tetrahedron_volume": maximum_volume,
        "surface_prescreen": records,
        "remesh_attempts": attempts,
    }
    return best, search


# ---------------------------------------------------------------------------
# MMG3D helpers shared by the legacy collar code and active level-set method
# ---------------------------------------------------------------------------

CAP_REFERENCE = 2
ANATOMICAL_WALL_REFERENCE = 1
LEVELSET_INPUT_REFERENCE = 1
LEVELSET_DISCARD_REFERENCE = 101
LEVELSET_RETAINED_REFERENCE = 102


def resolve_mmg3d_executable(requested: str) -> tuple[str, str]:
    """Resolve MMG3D once, before any expensive shooting is started."""
    candidates = [requested] if requested != "auto" else ["mmg3d_O3", "mmg3d"]
    resolved: str | None = None
    for candidate in candidates:
        direct = Path(candidate).expanduser()
        if direct.is_file():
            resolved = str(direct.resolve())
            break
        discovered = shutil.which(candidate)
        if discovered:
            resolved = str(Path(discovered).resolve())
            break
    if resolved is None:
        raise FileNotFoundError(
            "MMG3D was not found. Install it with "
            "`conda install -c conda-forge mmgsuite`, then confirm that "
            "`mmg3d_O3 -h` works. Alternatively pass --mmg3d-executable PATH."
        )
    try:
        probe = subprocess.run(
            [resolved, "-h"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not execute MMG3D at {resolved}: {exc}") from exc
    if probe.returncode < 0:
        raise RuntimeError(
            f"MMG3D help probe {_returncode_description(probe.returncode)}"
        )
    version_text = _bounded_process_text(probe.stdout, 2000)
    return resolved, version_text


def tetrahedral_face_topology(tetrahedra: np.ndarray) -> dict[str, np.ndarray]:
    """Return oriented boundary faces, owners, and face-adjacent tet pairs."""
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("Tetrahedral connectivity must have shape (N, 4)")
    # For a positively oriented [a,b,c,d] tetrahedron these local face orders
    # point outward. MMG can repair orientation, but supplying a coherent PLC
    # makes its analysis deterministic and easier to diagnose.
    local_faces = np.asarray(
        ([1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]), dtype=np.int64
    )
    faces = tetrahedra[:, local_faces].reshape(-1, 3)
    owners = np.repeat(np.arange(len(tetrahedra), dtype=np.int64), 4)
    keys = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise RuntimeError("Tetrahedral mesh has a non-manifold face")
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64))
    )
    representative = order[starts]
    boundary_groups = np.flatnonzero(counts == 1)
    boundary_indices = representative[boundary_groups]
    paired_groups = np.flatnonzero(counts == 2)
    first = order[starts[paired_groups]]
    second = order[starts[paired_groups] + 1]
    adjacency = np.column_stack((owners[first], owners[second])).astype(
        np.int64, copy=False
    )
    return {
        "boundary_faces": np.ascontiguousarray(faces[boundary_indices]),
        "boundary_owners": np.ascontiguousarray(owners[boundary_indices]),
        "adjacency": np.ascontiguousarray(adjacency),
    }


def local_collar_constraints(
    clipped: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    offset: float,
    retained_sign: int,
    collar_width: float,
    collar_rings: int,
    flat_tolerance: float,
) -> dict[str, Any]:
    """Build the MMG entities that freeze everything outside the basal collar."""
    points = np.asarray(
        vtk_to_numpy(clipped.GetPoints().GetData()), dtype=np.float64
    )
    tetrahedra = tetra_connectivity(clipped)
    topology = tetrahedral_face_topology(tetrahedra)
    boundary_faces = topology["boundary_faces"]
    boundary_owners = topology["boundary_owners"]
    adjacency = topology["adjacency"]
    coordinate = coordinates_along_normal(points, normal)
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    cap_tolerance = max(float(flat_tolerance) * 10.0, diagonal * 1e-9, 1e-8)
    residual = coordinate - float(offset)
    cap_face = np.all(np.abs(residual[boundary_faces]) <= cap_tolerance, axis=1)
    if np.count_nonzero(cap_face) == 0:
        raise RuntimeError("Directly clipped volume has no planar cap triangles")

    retained_depth = int(retained_sign) * residual
    if float(np.min(retained_depth)) < -cap_tolerance:
        raise RuntimeError("Directly clipped volume contains points on the removed side")
    tetra_depth = np.min(retained_depth[tetrahedra], axis=1)
    collar = tetra_depth <= float(collar_width) + cap_tolerance
    collar[boundary_owners[cap_face]] = True
    for _ in range(int(collar_rings)):
        if len(adjacency) == 0:
            break
        first = adjacency[:, 0]
        second = adjacency[:, 1]
        crossing = np.logical_xor(collar[first], collar[second])
        if not np.any(crossing):
            break
        grown = collar.copy()
        grown[first[crossing]] = True
        grown[second[crossing]] = True
        collar = grown
    frozen = ~collar
    if not np.any(collar):
        raise RuntimeError("The editable basal collar is empty")
    if not np.any(frozen):
        raise RuntimeError(
            "The basal collar covers the entire heart. Reduce --collar-width "
            "or --collar-rings so non-basal anatomy can be frozen."
        )

    required_triangles = np.flatnonzero(~cap_face).astype(np.int64)
    required_tetrahedra = np.flatnonzero(frozen).astype(np.int64)
    required_vertex_parts = [boundary_faces[~cap_face].ravel()]
    if len(required_tetrahedra):
        required_vertex_parts.append(tetrahedra[required_tetrahedra].ravel())
    required_vertices = np.unique(np.concatenate(required_vertex_parts)).astype(
        np.int64
    )
    triangle_references = np.where(
        cap_face, CAP_REFERENCE, ANATOMICAL_WALL_REFERENCE
    ).astype(np.int64)
    return {
        "points": points,
        "tetrahedra": tetrahedra,
        "boundary_faces": boundary_faces,
        "triangle_references": triangle_references,
        "cap_face": cap_face,
        "required_vertices": required_vertices,
        "required_triangles": required_triangles,
        "required_tetrahedra": required_tetrahedra,
        "frozen_tetrahedron_mask": frozen,
        "report": {
            "collar_width_mm": float(collar_width),
            "adjacency_rings": int(collar_rings),
            "editable_tetrahedra": int(np.count_nonzero(collar)),
            "frozen_tetrahedra": int(np.count_nonzero(frozen)),
            "editable_fraction": float(np.mean(collar)),
            "required_vertices": int(len(required_vertices)),
            "required_nonbasal_triangles": int(len(required_triangles)),
            "cap_triangles_before_mmg": int(np.count_nonzero(cap_face)),
            "cap_classification_tolerance": float(cap_tolerance),
        },
    }


def write_medit_mesh(path: Path, constraints: dict[str, Any]) -> None:
    """Write an ASCII Medit mesh with explicit MMG required entities."""
    points = np.asarray(constraints["points"], dtype=np.float64)
    triangles = np.asarray(constraints["boundary_faces"], dtype=np.int64)
    triangle_references = np.asarray(
        constraints["triangle_references"], dtype=np.int64
    )
    tetrahedra = np.asarray(constraints["tetrahedra"], dtype=np.int64)
    required_vertices = np.asarray(
        constraints["required_vertices"], dtype=np.int64
    )
    required_triangles = np.asarray(
        constraints["required_triangles"], dtype=np.int64
    )
    required_tetrahedra = np.asarray(
        constraints["required_tetrahedra"], dtype=np.int64
    )
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("MeshVersionFormatted 2\n\nDimension 3\n\n")
        stream.write(f"Vertices\n{len(points)}\n")
        point_rows = np.column_stack(
            (points, np.zeros(len(points), dtype=np.int64))
        )
        np.savetxt(stream, point_rows, fmt="%.17g %.17g %.17g %d")
        stream.write(f"\nTriangles\n{len(triangles)}\n")
        np.savetxt(
            stream,
            np.column_stack((triangles + 1, triangle_references)),
            fmt="%d %d %d %d",
        )
        stream.write(f"\nTetrahedra\n{len(tetrahedra)}\n")
        np.savetxt(
            stream,
            np.column_stack(
                (tetrahedra + 1, np.ones(len(tetrahedra), dtype=np.int64))
            ),
            fmt="%d %d %d %d %d",
        )
        stream.write(f"\nRequiredVertices\n{len(required_vertices)}\n")
        if len(required_vertices):
            np.savetxt(stream, required_vertices + 1, fmt="%d")
        stream.write(f"\nRequiredTriangles\n{len(required_triangles)}\n")
        if len(required_triangles):
            np.savetxt(stream, required_triangles + 1, fmt="%d")
        stream.write(f"\nRequiredTetrahedra\n{len(required_tetrahedra)}\n")
        if len(required_tetrahedra):
            np.savetxt(stream, required_tetrahedra + 1, fmt="%d")
        stream.write("\nEnd\n")


def read_medit_mesh(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the geometry sections needed from an ASCII MMG .mesh output."""
    if not path.is_file():
        raise FileNotFoundError(f"MMG3D did not create {path.name}")
    clean: list[str] = []
    for raw in path.read_text(encoding="ascii", errors="strict").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            clean.append(line)

    def section(name: str, width: int) -> list[list[str]]:
        index = next(
            (i for i, value in enumerate(clean) if value.lower() == name.lower()),
            None,
        )
        if index is None or index + 1 >= len(clean):
            if name == "Triangles":
                return []
            raise ValueError(f"MMG output has no {name} section")
        try:
            count = int(clean[index + 1].split()[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"MMG output has an invalid {name} count") from exc
        rows: list[list[str]] = []
        for line in clean[index + 2 : index + 2 + count]:
            values = line.split()
            if len(values) < width:
                raise ValueError(f"MMG output has a truncated {name} row")
            rows.append(values[:width])
        if len(rows) != count:
            raise ValueError(f"MMG output has a truncated {name} section")
        return rows

    vertex_rows = section("Vertices", 4)
    tetrahedron_rows = section("Tetrahedra", 5)
    triangle_rows = section("Triangles", 4)
    points = np.asarray(
        [[float(value) for value in row[:3]] for row in vertex_rows],
        dtype=np.float64,
    )
    tetrahedra = np.asarray(
        [[int(value) - 1 for value in row[:4]] for row in tetrahedron_rows],
        dtype=np.int64,
    )
    if triangle_rows:
        triangles = np.asarray(
            [[int(value) - 1 for value in row[:3]] for row in triangle_rows],
            dtype=np.int64,
        )
        triangle_references = np.asarray(
            [int(row[3]) for row in triangle_rows], dtype=np.int64
        )
    else:
        triangles = np.empty((0, 3), dtype=np.int64)
        triangle_references = np.empty(0, dtype=np.int64)
    if not np.isfinite(points).all():
        raise FloatingPointError("MMG3D returned non-finite coordinates")
    if len(points) == 0 or len(tetrahedra) == 0:
        raise RuntimeError("MMG3D returned an empty tetrahedral mesh")
    return points, tetrahedra, triangles, triangle_references


def project_referenced_cap(
    grid: vtk.vtkUnstructuredGrid,
    cap_vertex_ids: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> dict[str, Any]:
    """Make only MMG's reference-2 cap vertices exactly coplanar."""
    cap_vertex_ids = np.unique(np.asarray(cap_vertex_ids, dtype=np.int64))
    points = np.asarray(
        vtk_to_numpy(grid.GetPoints().GetData()), dtype=np.float64
    ).copy()
    if len(cap_vertex_ids) < 3:
        raise RuntimeError("MMG3D output has fewer than three referenced cap vertices")
    if cap_vertex_ids.min() < 0 or cap_vertex_ids.max() >= len(points):
        raise ValueError("MMG3D cap references contain invalid point indices")
    residual = coordinates_along_normal(points[cap_vertex_ids], normal) - float(
        offset
    )
    before = float(np.max(np.abs(residual)))
    points[cap_vertex_ids] -= residual[:, None] * np.asarray(normal)[None, :]
    if not np.isfinite(points).all():
        raise FloatingPointError("Exact cap projection produced non-finite points")
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_to_vtk(
            np.ascontiguousarray(points), deep=True, array_type=vtk.VTK_DOUBLE
        )
    )
    grid.SetPoints(vtk_points)
    after = float(
        np.max(
            np.abs(
                coordinates_along_normal(points[cap_vertex_ids], normal)
                - float(offset)
            )
        )
    )
    return {
        "method": "MMG_triangle_reference_2_normal_projection",
        "projected_vertices": int(len(cap_vertex_ids)),
        "maximum_residual_before_projection": before,
        "maximum_residual_after_projection": after,
    }


def verify_required_tetrahedra_preserved(
    input_grid: vtk.vtkUnstructuredGrid,
    output_grid: vtk.vtkUnstructuredGrid,
    required_tetrahedra: np.ndarray,
    required_vertices: np.ndarray,
    required_triangle_vertices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Verify required vertices, boundary triangles, and tetrahedra independently."""
    required_tetrahedra = np.asarray(required_tetrahedra, dtype=np.int64)
    required_vertices = np.asarray(required_vertices, dtype=np.int64)
    required_triangle_vertices = np.asarray(
        (
            np.empty((0, 3), dtype=np.int64)
            if required_triangle_vertices is None
            else required_triangle_vertices
        ),
        dtype=np.int64,
    ).reshape(-1, 3)
    input_points = np.asarray(
        vtk_to_numpy(input_grid.GetPoints().GetData()), dtype=np.float64
    )
    input_tets = tetra_connectivity(input_grid)
    output_points = np.asarray(
        vtk_to_numpy(output_grid.GetPoints().GetData()), dtype=np.float64
    )
    output_tets = tetra_connectivity(output_grid)
    frozen_points = np.unique(
        np.concatenate(
            (
                input_tets[required_tetrahedra].ravel(),
                required_vertices,
                required_triangle_vertices.ravel(),
            )
        )
    )
    if len(frozen_points) == 0:
        raise RuntimeError("MMG preservation check received no required entities")
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(output_grid)
    locator.BuildLocator()
    mapped = np.fromiter(
        (locator.FindClosestPoint(input_points[index]) for index in frozen_points),
        dtype=np.int64,
        count=len(frozen_points),
    )
    distance = np.linalg.norm(output_points[mapped] - input_points[frozen_points], axis=1)
    diagonal = max(float(np.linalg.norm(np.ptp(input_points, axis=0))), 1.0)
    tolerance = max(diagonal * 1e-10, 1e-9)
    input_to_output = np.full(len(input_points), -1, dtype=np.int64)
    input_to_output[frozen_points] = mapped
    mapped_tets = np.sort(input_to_output[input_tets[required_tetrahedra]], axis=1)
    output_tet_set = {
        tuple(int(value) for value in row) for row in np.sort(output_tets, axis=1)
    }
    preserved = np.fromiter(
        (
            tuple(int(value) for value in row) in output_tet_set
            for row in mapped_tets
        ),
        dtype=bool,
        count=len(mapped_tets),
    )
    mapped_triangles = np.sort(
        input_to_output[required_triangle_vertices], axis=1
    )
    output_boundary = tetrahedral_face_topology(output_tets)["boundary_faces"]
    output_triangle_set = {
        tuple(int(value) for value in row)
        for row in np.sort(output_boundary, axis=1)
    }
    preserved_triangles = np.fromiter(
        (
            tuple(int(value) for value in row) in output_triangle_set
            for row in mapped_triangles
        ),
        dtype=bool,
        count=len(mapped_triangles),
    )
    unique_mapping = len(np.unique(mapped)) == len(mapped)
    maximum_displacement = float(np.max(distance))
    required_boundary_vertices = np.unique(required_triangle_vertices.ravel())
    if len(required_boundary_vertices):
        boundary_output_ids = input_to_output[required_boundary_vertices]
        if np.any(boundary_output_ids < 0):
            raise RuntimeError(
                "Protected boundary vertices were omitted from the MMG mapping"
            )
        boundary_distance = np.linalg.norm(
            output_points[boundary_output_ids]
            - input_points[required_boundary_vertices],
            axis=1,
        )
        maximum_boundary_displacement = float(np.max(boundary_distance))
        boundary_mapping_is_one_to_one = bool(
            len(np.unique(boundary_output_ids)) == len(boundary_output_ids)
        )
    else:
        maximum_boundary_displacement = 0.0
        boundary_mapping_is_one_to_one = True
    protected_geometry_preserved = bool(
        boundary_mapping_is_one_to_one
        and maximum_boundary_displacement <= tolerance
        and np.all(preserved_triangles)
    )
    return {
        "required_tetrahedra": int(len(required_tetrahedra)),
        "explicitly_required_vertices": int(len(required_vertices)),
        "verified_required_vertices": int(len(frozen_points)),
        "required_boundary_triangles": int(len(required_triangle_vertices)),
        "maximum_required_vertex_displacement": maximum_displacement,
        "maximum_required_boundary_vertex_displacement": (
            maximum_boundary_displacement
        ),
        "coordinate_tolerance": float(tolerance),
        "required_vertex_mapping_is_one_to_one": bool(unique_mapping),
        "required_boundary_vertex_mapping_is_one_to_one": (
            boundary_mapping_is_one_to_one
        ),
        "preserved_required_tetrahedra": int(np.count_nonzero(preserved)),
        "all_required_tetrahedra_preserved": bool(np.all(preserved)),
        "required_tetrahedron_connectivity_is_diagnostic": True,
        "preserved_required_boundary_triangles": int(
            np.count_nonzero(preserved_triangles)
        ),
        "all_required_boundary_triangles_preserved": bool(
            np.all(preserved_triangles)
        ),
        "protected_geometry_preserved": protected_geometry_preserved,
        # Interior node relocation/connectivity changes are allowed because
        # they improve element quality without changing anatomical geometry.
        # Anatomical preservation is defined by fixed protected boundary
        # coordinates and an unchanged protected boundary triangulation.
        "passed": protected_geometry_preserved,
    }


def run_mmg3d_collar_mode(
    clipped: vtk.vtkUnstructuredGrid,
    constraints: dict[str, Any],
    normal: np.ndarray,
    offset: float,
    executable: str,
    mode: str,
    args: argparse.Namespace,
) -> tuple[vtk.vtkUnstructuredGrid, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run one isolated MMG3D mode and validate all serialized geometry."""
    with tempfile.TemporaryDirectory(prefix="ct55_mmg3d_") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "collar_input.mesh"
        output_path = temporary_path / "collar_output.mesh"
        write_medit_mesh(input_path, constraints)
        command = [
            executable,
            str(input_path),
            "-out",
            str(output_path),
            "-optim",
            "-hausd",
            f"{args.mmg_hausdorff:.17g}",
            "-hgrad",
            f"{args.mmg_gradation:.17g}",
            "-v",
            "-1",
        ]
        if mode == "nofem_fallback":
            command.append("-nofem")
        elif mode != "standard":
            raise ValueError(f"Unknown MMG3D mode: {mode}")
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=temporary_path,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(args.mmg_timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            timeout_stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout
            )
            timeout_stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr
            )
            raise RuntimeError(
                f"MMG3D {mode} timed out after {duration:.1f} s "
                f"(limit {args.mmg_timeout_seconds:g} s); "
                f"stdout={_bounded_process_text(timeout_stdout)!r}; "
                f"stderr={_bounded_process_text(timeout_stderr)!r}"
            ) from None
        duration = time.monotonic() - started
        stdout = _bounded_process_text(completed.stdout)
        stderr = _bounded_process_text(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"MMG3D {mode} {_returncode_description(completed.returncode)} "
                f"after {duration:.1f} s; stdout={stdout!r}; stderr={stderr!r}"
            )
        points, tetrahedra, triangles, triangle_references = read_medit_mesh(
            output_path
        )
        grid = build_tetrahedral_grid(points, tetrahedra)
        cap_triangles = triangles[triangle_references == CAP_REFERENCE]
        if len(cap_triangles) == 0:
            raise RuntimeError(
                "MMG3D output lost the planar-cap triangle reference"
            )
        cap_projection = project_referenced_cap(
            grid, cap_triangles.ravel(), normal, offset
        )
        frozen = verify_required_tetrahedra_preserved(
            clipped,
            grid,
            constraints["required_tetrahedra"],
            constraints["required_vertices"],
            constraints["boundary_faces"][constraints["required_triangles"]],
        )
        report = {
            "backend": "MMG3D_required_entity_local_optimization",
            "executable": executable,
            "mode": mode,
            "command": command,
            "wall_time_seconds": float(duration),
            "returncode": int(completed.returncode),
            "stdout": stdout,
            "stderr": stderr,
            "input_points": int(clipped.GetNumberOfPoints()),
            "input_tetrahedra": int(clipped.GetNumberOfCells()),
            "output_points": int(grid.GetNumberOfPoints()),
            "output_tetrahedra": int(grid.GetNumberOfCells()),
            "hausdorff_tolerance": float(args.mmg_hausdorff),
            "gradation": float(args.mmg_gradation),
            "timeout_seconds": float(args.mmg_timeout_seconds),
        }
        return grid, report, cap_projection, frozen


def local_remesh_bundle_score(bundle: dict[str, Any]) -> tuple[float, ...]:
    quality = bundle["volume_quality"]
    return (
        float(bool(bundle["passed"])),
        float(quality["minimum_scaled_jacobian"] or -1.0),
        float(quality["p01_scaled_jacobian"] or -1.0),
        float(quality["minimum_dihedral_angle"] or -1.0),
        -float(bundle["candidate"]["extra_inward_depth"]),
    )


def evaluate_local_mmg_output(
    grid: vtk.vtkUnstructuredGrid,
    direct_boundary: vtk.vtkPolyData,
    direct_surface_quality: dict[str, Any],
    direct_cap_topology: dict[str, Any],
    raw_surface: np.ndarray,
    template_faces: np.ndarray,
    template_object_labels: np.ndarray,
    shot_grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    offset: float,
    candidate: dict[str, Any],
    mmg_report: dict[str, Any],
    cap_projection: dict[str, Any],
    frozen_report: dict[str, Any],
    collar_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = transfer_volume_metadata(shot_grid, grid)
    final_surface, basal_face, basal_vertex = boundary_from_volume(
        grid, normal, offset
    )
    anatomical_labels = add_anatomical_object_labels(
        final_surface,
        raw_surface,
        template_faces,
        template_object_labels,
    )
    volume_quality = clipped_volume_quality(grid)
    surface_quality = extracted_surface_quality(
        final_surface, basal_face, basal_vertex, normal, offset
    )
    final_points, final_faces = polydata_points_and_faces(final_surface)
    final_cap_topology = basal_cap_topology(final_faces, basal_face)
    fidelity = bidirectional_surface_fidelity(direct_boundary, final_surface)
    diagonal = max(float(np.linalg.norm(np.ptp(final_points, axis=0))), 1.0)
    fidelity_tolerance = max(
        diagonal * 1e-6, float(args.mmg_hausdorff) * 2.0, 1e-5
    )
    # Label 0 is the artificial basal closure, not a fourth anatomical
    # object.  The three positive labels are the outer wall, LV and RV.
    expected_objects = sorted(
        int(value)
        for value in np.unique(template_object_labels)
        if int(value) > 0
    )
    output_anatomical_objects = sorted(
        int(value)
        for value in anatomical_labels["output_objects"]
        if int(value) > 0
    )
    checks = {
        "positive_tetrahedra": volume_quality["nonpositive_scaled_jacobians"] == 0,
        "finite_tetrahedral_quality": volume_quality[
            "nonfinite_scaled_jacobians"
        ]
        == 0,
        "nondegenerate_tetrahedra": volume_quality["degenerate_tetrahedra"] == 0,
        "minimum_scaled_jacobian": volume_quality["minimum_scaled_jacobian"]
        is not None
        and volume_quality["minimum_scaled_jacobian"]
        >= args.minimum_scaled_jacobian,
        "p01_scaled_jacobian": volume_quality["p01_scaled_jacobian"] is not None
        and volume_quality["p01_scaled_jacobian"]
        >= args.minimum_p01_scaled_jacobian,
        "minimum_dihedral_angle": volume_quality["minimum_dihedral_angle"]
        is not None
        and volume_quality["minimum_dihedral_angle"]
        >= args.minimum_dihedral_angle,
        "flat_base": surface_quality["maximum_basal_plane_error"]
        <= args.flat_tolerance,
        "closed_surface": surface_quality["boundary_edges"] == 0,
        "manifold_surface": surface_quality["nonmanifold_edges"] == 0,
        "nondegenerate_surface": surface_quality["degenerate_triangles"] == 0,
        "expected_outer_lv_rv_contours": (
            args.expected_basal_contours == 0
            or final_cap_topology["contour_loops"] == args.expected_basal_contours
        ),
        "direct_clip_had_expected_contours": (
            args.expected_basal_contours == 0
            or direct_cap_topology["contour_loops"] == args.expected_basal_contours
        ),
        "surface_matches_direct_clip": fidelity[
            "maximum_bidirectional_distance"
        ]
        <= fidelity_tolerance,
        "all_anatomical_objects_preserved": output_anatomical_objects
        == expected_objects,
        "basal_cap_has_dedicated_zero_label": (
            0 not in anatomical_labels["source_objects"]
            or anatomical_labels["basal_triangles_assigned_zero"]
            == anatomical_labels["basal_triangles"]
        ),
        "all_required_tetrahedra_preserved": bool(frozen_report["passed"]),
    }
    return {
        "grid": grid,
        "surface": final_surface,
        "candidate": candidate,
        "mmg3d": mmg_report,
        "cap_projection": cap_projection,
        "frozen_region": frozen_report,
        "collar": collar_report,
        "volume_metadata": metadata,
        "anatomical_label_transfer": anatomical_labels,
        "clipped_surface_quality": direct_surface_quality,
        "clipped_cap_topology": direct_cap_topology,
        "volume_quality": volume_quality,
        "surface_quality": surface_quality,
        "final_cap_topology": final_cap_topology,
        "surface_fidelity": fidelity,
        "surface_fidelity_tolerance": fidelity_tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def remesh_one_local_collar_candidate(
    raw_surface: np.ndarray,
    template_faces: np.ndarray,
    template_object_labels: np.ndarray,
    shot_grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    retained_sign: int,
    candidate: dict[str, Any],
    executable: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Clip one plane, locally optimize its collar, and perform complete QA."""
    offset = float(candidate["offset"])
    clipped = clip_tetrahedral_volume(shot_grid, normal, offset, retained_sign)
    direct_projection = project_numerical_cap_residual(clipped, normal, offset)
    direct_volume_quality = clipped_volume_quality(clipped)
    if direct_volume_quality["nonfinite_scaled_jacobians"] != 0:
        raise RuntimeError("Direct clip produced non-finite tetrahedral quality")
    if direct_volume_quality["nonpositive_scaled_jacobians"] != 0:
        raise RuntimeError("Direct clip produced non-positive tetrahedra")
    if direct_volume_quality["degenerate_tetrahedra"] != 0:
        raise RuntimeError("Direct clip produced zero-volume tetrahedra")
    direct_boundary, direct_basal_face, direct_basal_vertex = boundary_from_volume(
        clipped, normal, offset
    )
    direct_surface_quality = extracted_surface_quality(
        direct_boundary,
        direct_basal_face,
        direct_basal_vertex,
        normal,
        offset,
    )
    direct_points, direct_faces = polydata_points_and_faces(direct_boundary)
    direct_cap_topology = basal_cap_topology(direct_faces, direct_basal_face)
    if direct_surface_quality["boundary_edges"] != 0:
        raise RuntimeError("Directly clipped volume has an open boundary")
    if direct_surface_quality["nonmanifold_edges"] != 0:
        raise RuntimeError("Directly clipped volume has a non-manifold boundary")
    if (
        args.expected_basal_contours > 0
        and direct_cap_topology["contour_loops"] != args.expected_basal_contours
    ):
        raise RuntimeError(
            "Directly clipped cap does not contain the expected outer/LV/RV loops"
        )
    constraints = local_collar_constraints(
        clipped,
        normal,
        offset,
        retained_sign,
        args.collar_width,
        args.collar_rings,
        args.flat_tolerance,
    )
    modes = ["standard"]
    if args.mmg_nofem_fallback:
        modes.append("nofem_fallback")
    attempts: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    for mode in modes:
        try:
            grid, mmg_report, cap_projection, frozen_report = (
                run_mmg3d_collar_mode(
                    clipped,
                    constraints,
                    normal,
                    offset,
                    executable,
                    mode,
                    args,
                )
            )
            bundle = evaluate_local_mmg_output(
                grid,
                direct_boundary,
                direct_surface_quality,
                direct_cap_topology,
                raw_surface,
                template_faces,
                template_object_labels,
                shot_grid,
                normal,
                offset,
                candidate,
                mmg_report,
                cap_projection,
                frozen_report,
                constraints["report"],
                args,
            )
            bundle["direct_clip_projection"] = direct_projection
            bundle["direct_clip_volume_quality"] = direct_volume_quality
            bundles.append(bundle)
            attempts.append(
                {
                    "mode": mode,
                    "passed": bool(bundle["passed"]),
                    "checks": bundle["checks"],
                    "mmg3d": mmg_report,
                    "volume_quality": bundle["volume_quality"],
                    "frozen_region": frozen_report,
                }
            )
            if bundle["passed"]:
                bundle["mmg_mode_attempts"] = attempts
                return bundle
        except Exception as exc:
            attempts.append({"mode": mode, "passed": False, "error": str(exc)})
    if not bundles:
        reasons = " | ".join(
            f"{attempt['mode']}: {attempt.get('error', 'failed QA')}"
            for attempt in attempts
        )
        raise RuntimeError("Local MMG3D collar optimization failed: " + reasons)
    best = max(bundles, key=local_remesh_bundle_score)
    best["mmg_mode_attempts"] = attempts
    return best


def quality_driven_local_collar_remesh(
    raw_surface: np.ndarray,
    template: dict[str, Any],
    shot_grid: vtk.vtkUnstructuredGrid,
    shot_volume_points: np.ndarray,
    normal: np.ndarray,
    plane: dict[str, Any],
    executable: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return the shallowest locally remeshed cut that passes all QA checks."""
    candidates = candidate_cut_offsets(
        shot_volume_points,
        plane,
        float(plane["ct55_offset"]),
        args.maximum_plane_offset_shift,
        args.cut_search_distance,
        args.cut_search_step,
        args.cut_search_maximum_candidates,
    )
    retained_sign = int(plane["retained_implicit_sign"])
    template_faces = np.asarray(template["faces"], dtype=np.int64)
    object_labels = np.asarray(template["object_labels"], dtype=np.int32)
    prescreen: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        record: dict[str, Any] = {
            "candidate": index,
            "offset": float(candidate["offset"]),
            "shift_from_ct55": float(candidate["offset"])
            - float(plane["ct55_offset"]),
            "extra_inward_depth": float(candidate["extra_inward_depth"]),
            "nearest_volume_vertex_distance": float(
                candidate["nearest_volume_vertex_distance"]
            ),
        }
        # Keep the historical field name for downstream report readers.
        record["nearest_surface_vertex_distance"] = record[
            "nearest_volume_vertex_distance"
        ]
        try:
            _, basal_face, _, surface_quality, topology = clip_and_cap_surface(
                raw_surface,
                template_faces,
                normal,
                float(candidate["offset"]),
                retained_sign,
                args.flat_tolerance,
            )
            minimum_clearance = float(plane["cut_clearance"]) + float(
                candidate["extra_inward_depth"]
            )
            checks = {
                "complete_old_base_removed": bool(
                    plane["complete_cut_within_bound"]
                )
                and minimum_clearance >= args.cut_clearance - 1e-10,
                "flat_cap": surface_quality["maximum_basal_plane_error"]
                <= args.flat_tolerance,
                "closed_surface": surface_quality["boundary_edges"] == 0,
                "manifold_surface": surface_quality["nonmanifold_edges"] == 0,
                "nondegenerate_surface": surface_quality[
                    "degenerate_triangles"
                ]
                == 0,
                "retained_side": surface_quality[
                    "maximum_retained_side_violation"
                ]
                <= max(args.flat_tolerance, 1e-8),
                "cap_has_triangles": int(np.count_nonzero(basal_face)) > 0,
                "cap_contours_are_loops": topology[
                    "every_boundary_vertex_has_degree_two"
                ],
                "expected_outer_lv_rv_contours": (
                    args.expected_basal_contours == 0
                    or topology["contour_loops"] == args.expected_basal_contours
                ),
            }
            record.update(
                {
                    "minimum_old_base_clearance": minimum_clearance,
                    "surface_quality": surface_quality,
                    "cap_topology": topology,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
            if record["passed"]:
                valid.append(record)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            record.update({"passed": False, "error": str(exc)})
        prescreen.append(record)

    # Try one candidate per half-step neighbourhood. This avoids spending two
    # MMG calls on numerically almost identical plane proposals while retaining
    # strict shallow-to-deep selection.
    valid.sort(key=lambda value: value["extra_inward_depth"])
    selected_records: list[dict[str, Any]] = []
    minimum_separation = max(float(args.cut_search_step) * 0.45, 1e-6)
    for record in valid:
        if (
            not selected_records
            or record["extra_inward_depth"]
            - selected_records[-1]["extra_inward_depth"]
            >= minimum_separation
        ):
            selected_records.append(record)
        elif (
            record["nearest_volume_vertex_distance"]
            > selected_records[-1]["nearest_volume_vertex_distance"]
        ):
            selected_records[-1] = record
        if len(selected_records) >= args.remesh_candidates:
            break

    attempts: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for remesh_index, record in enumerate(selected_records, start=1):
        print(
            f"  local-collar candidate {remesh_index}/{len(selected_records)}: "
            f"cut depth={record['extra_inward_depth']:.4g} mm",
            flush=True,
        )
        attempt: dict[str, Any] = {
            "candidate": int(record["candidate"]),
            "offset": float(record["offset"]),
            "shift_from_ct55": float(record["shift_from_ct55"]),
            "extra_inward_depth": float(record["extra_inward_depth"]),
            "nearest_volume_vertex_distance": float(
                record["nearest_volume_vertex_distance"]
            ),
        }
        try:
            bundle = remesh_one_local_collar_candidate(
                raw_surface,
                template_faces,
                object_labels,
                shot_grid,
                normal,
                retained_sign,
                record,
                executable,
                args,
            )
            bundles.append(bundle)
            attempt.update(
                {
                    "passed": bool(bundle["passed"]),
                    "checks": bundle["checks"],
                    "mmg3d": bundle["mmg3d"],
                    "mmg_mode_attempts": bundle["mmg_mode_attempts"],
                    "collar": bundle["collar"],
                    "frozen_region": bundle["frozen_region"],
                    "volume_quality": bundle["volume_quality"],
                    "surface_quality": bundle["surface_quality"],
                    "cap_topology": bundle["final_cap_topology"],
                }
            )
            if bundle["passed"]:
                selected = bundle
                attempts.append(attempt)
                break
        except Exception as exc:
            attempt.update({"passed": False, "error": str(exc)})
            print(
                "    failed safely: " + _bounded_process_text(str(exc), 1200),
                flush=True,
            )
        attempts.append(attempt)
    if selected is None and bundles:
        selected = max(bundles, key=local_remesh_bundle_score)
    search = {
        "method": "exact_volume_clip_then_required_entity_local_MMG3D_collar",
        "selection_rule": (
            "shallowest fully passing candidate; best-quality attempted mesh "
            "is returned but remains rejected when none passes"
        ),
        "candidate_planes_constructed": len(candidates),
        "surface_valid_candidates": len(valid),
        "remesh_candidates_requested": int(args.remesh_candidates),
        "remesh_candidates_attempted": len(attempts),
        "passing_remeshes": int(sum(bool(item.get("passed")) for item in attempts)),
        "passing_candidate_found": bool(
            selected is not None and selected["passed"]
        ),
        "surface_prescreen": prescreen,
        "remesh_attempts": attempts,
    }
    return selected, search


# ---------------------------------------------------------------------------
# Implicit MMG3D level-set cut (the active generation method)
# ---------------------------------------------------------------------------


def tetrahedral_edges(tetrahedra: np.ndarray) -> np.ndarray:
    """Return unique, undirected edges from tetrahedral connectivity."""
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    local_edges = np.asarray(
        ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]),
        dtype=np.int64,
    )
    edges = np.sort(tetrahedra[:, local_edges].reshape(-1, 2), axis=1)
    return np.unique(edges, axis=0)


def positively_oriented_tetrahedra(
    points: np.ndarray, tetrahedra: np.ndarray
) -> np.ndarray:
    """Return equivalent connectivity with positive signed volumes."""
    points = np.asarray(points, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    signed = signed_tetra_volumes(points, tetrahedra)
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    tolerance = max(diagonal**3 * 1e-15, 1e-15)
    if not np.isfinite(signed).all():
        raise FloatingPointError("Input tetrahedra contain non-finite volumes")
    if np.any(np.abs(signed) <= tolerance):
        raise RuntimeError("The uncut shot volume contains zero-volume tetrahedra")
    oriented = tetrahedra.copy()
    negative = signed < 0
    if np.any(negative):
        original = oriented[negative].copy()
        oriented[negative, 0] = original[:, 1]
        oriented[negative, 1] = original[:, 0]
    if np.any(signed_tetra_volumes(points, oriented) <= tolerance):
        raise RuntimeError("Could not orient the uncut shot tetrahedra positively")
    return oriented


def grow_tetrahedron_selection(
    selection: np.ndarray, adjacency: np.ndarray, rings: int
) -> np.ndarray:
    """Grow a tetrahedron mask by face-adjacency rings."""
    grown = np.asarray(selection, dtype=bool).copy()
    adjacency = np.asarray(adjacency, dtype=np.int64)
    for _ in range(int(rings)):
        if len(adjacency) == 0:
            break
        first = adjacency[:, 0]
        second = adjacency[:, 1]
        boundary = np.logical_xor(grown[first], grown[second])
        if not np.any(boundary):
            break
        updated = grown.copy()
        updated[first[boundary]] = True
        updated[second[boundary]] = True
        grown = updated
    return grown


def levelset_protection_constraints(
    shot_grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    offset: float,
    retained_sign: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Protect the uncut anatomy while releasing the future cap-wall seam."""
    points = np.asarray(
        vtk_to_numpy(shot_grid.GetPoints().GetData()), dtype=np.float64
    )
    tetrahedra = positively_oriented_tetrahedra(
        points, tetra_connectivity(shot_grid)
    )
    topology = tetrahedral_face_topology(tetrahedra)
    boundary_faces = topology["boundary_faces"]
    adjacency = topology["adjacency"]
    coordinate = coordinates_along_normal(points, normal)
    phi = int(retained_sign) * (coordinate - float(offset))
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    zero_tolerance = max(diagonal * 1e-12, 1e-10)
    tet_phi = phi[tetrahedra]
    minimum_phi = np.min(tet_phi, axis=1)
    maximum_phi = np.max(tet_phi, axis=1)
    crossing = np.logical_and(
        minimum_phi < -zero_tolerance, maximum_phi > zero_tolerance
    )
    if not np.any(crossing):
        raise RuntimeError("Candidate plane does not cross the shot tetrahedral mesh")
    if not np.any(maximum_phi < -zero_tolerance):
        raise RuntimeError("Candidate plane leaves no removable tetrahedral domain")
    if not np.any(minimum_phi > zero_tolerance):
        raise RuntimeError("Candidate plane leaves no retained tetrahedral domain")

    crossing_edges = tetrahedral_edges(tetrahedra[crossing])
    crossing_lengths = np.linalg.norm(
        points[crossing_edges[:, 1]] - points[crossing_edges[:, 0]], axis=1
    )
    positive_lengths = crossing_lengths[
        np.logical_and(np.isfinite(crossing_lengths), crossing_lengths > 0)
    ]
    if len(positive_lengths) == 0:
        raise RuntimeError("Cannot estimate the basal tetrahedral edge length")
    local_edge = float(np.median(positive_lengths))
    automatic_wall_release = min(
        float(args.collar_width) * 0.75,
        max(2.5 * local_edge, 10.0 * float(args.mmg_hausdorff)),
    )
    wall_release = (
        float(args.wall_release_width)
        if args.wall_release_width > 0
        else automatic_wall_release
    )
    wall_release = min(wall_release, float(args.collar_width) * 0.9)
    if wall_release <= 0:
        raise ValueError("The computed wall-release width is not positive")

    minimum_absolute_phi = np.min(np.abs(tet_phi), axis=1)
    editable = np.logical_or(
        crossing,
        minimum_absolute_phi <= float(args.collar_width) + zero_tolerance,
    )
    editable = grow_tetrahedron_selection(
        editable, adjacency, int(args.collar_rings)
    )
    frozen_retained = np.logical_and(
        ~editable, minimum_phi > zero_tolerance
    )
    if not np.any(frozen_retained):
        raise RuntimeError(
            "No retained tetrahedra remain outside the editable level-set collar"
        )

    boundary_minimum_phi = np.min(phi[boundary_faces], axis=1)
    required_triangle_mask = boundary_minimum_phi > wall_release
    required_triangles = np.flatnonzero(required_triangle_mask).astype(np.int64)
    required_tetrahedra = np.flatnonzero(frozen_retained).astype(np.int64)
    required_vertices = np.unique(
        np.concatenate(
            (
                boundary_faces[required_triangles].ravel(),
                tetrahedra[required_tetrahedra].ravel(),
            )
        )
    ).astype(np.int64)
    if len(required_triangles) == 0 or len(required_vertices) == 0:
        raise RuntimeError("The protected non-basal anatomical region is empty")

    hmin = max(0.35 * local_edge, diagonal * 1e-6)
    hmax = max(1.25 * local_edge, hmin * 1.01)
    return {
        "points": points,
        "tetrahedra": tetrahedra,
        "boundary_faces": boundary_faces,
        "triangle_references": np.full(
            len(boundary_faces),
            ANATOMICAL_WALL_REFERENCE,
            dtype=np.int64,
        ),
        "required_vertices": required_vertices,
        "required_triangles": required_triangles,
        "required_tetrahedra": required_tetrahedra,
        "phi": np.asarray(phi, dtype=np.float64),
        "crossing_tetrahedron_mask": crossing,
        "editable_tetrahedron_mask": editable,
        "frozen_tetrahedron_mask": frozen_retained,
        "report": {
            "method": "uncut_volume_affine_levelset_protected_collar",
            "levelset_definition": "phi=s*(n.x-d); retained domain phi>0",
            "zero_tolerance": float(zero_tolerance),
            "local_median_edge_length_mm": local_edge,
            "mmg_hmin_mm": float(hmin),
            "mmg_hmax_mm": float(hmax),
            "collar_width_mm": float(args.collar_width),
            "adjacency_rings": int(args.collar_rings),
            "wall_release_width_mm": float(wall_release),
            "wall_release_was_automatic": bool(args.wall_release_width == 0),
            "crossing_tetrahedra": int(np.count_nonzero(crossing)),
            "editable_tetrahedra": int(np.count_nonzero(editable)),
            "frozen_retained_tetrahedra": int(len(required_tetrahedra)),
            "required_nonbasal_triangles": int(len(required_triangles)),
            "required_vertices": int(len(required_vertices)),
        },
    }


def write_medit_scalar_solution(path: Path, values: np.ndarray) -> None:
    """Write one scalar per vertex in Medit .sol format."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Level-set values must be finite and non-empty")
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("MeshVersionFormatted 2\n\nDimension 3\n\n")
        stream.write(f"SolAtVertices\n{len(values)}\n1 1\n")
        np.savetxt(stream, values, fmt="%.17g")
        stream.write("\nEnd\n")


def write_mmg3d_levelset_parameters(path: Path) -> None:
    """Map the original material to explicit negative/positive LS domains."""
    path.write_text(
        "LSReferences\n"
        "1\n"
        f"{LEVELSET_INPUT_REFERENCE} "
        f"{LEVELSET_DISCARD_REFERENCE} "
        f"{LEVELSET_RETAINED_REFERENCE}\n",
        encoding="ascii",
    )


def read_medit_mesh_with_references(path: Path) -> dict[str, np.ndarray]:
    """Read geometry and element references from an ASCII Medit mesh."""
    if not path.is_file():
        raise FileNotFoundError(f"MMG3D did not create {path.name}")
    clean: list[str] = []
    for raw in path.read_text(encoding="ascii", errors="strict").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            clean.append(line)

    def section(name: str, width: int, optional: bool = False) -> list[list[str]]:
        index = next(
            (i for i, value in enumerate(clean) if value.lower() == name.lower()),
            None,
        )
        if index is None:
            if optional:
                return []
            raise ValueError(f"MMG output has no {name} section")
        if index + 1 >= len(clean):
            raise ValueError(f"MMG output has no {name} count")
        try:
            count = int(clean[index + 1].split()[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"MMG output has an invalid {name} count") from exc
        rows: list[list[str]] = []
        for line in clean[index + 2 : index + 2 + count]:
            values = line.split()
            if len(values) < width:
                raise ValueError(f"MMG output has a truncated {name} row")
            rows.append(values[:width])
        if len(rows) != count:
            raise ValueError(f"MMG output has a truncated {name} section")
        return rows

    vertex_rows = section("Vertices", 4)
    tetrahedron_rows = section("Tetrahedra", 5)
    triangle_rows = section("Triangles", 4, optional=True)
    points = np.asarray(
        [[float(value) for value in row[:3]] for row in vertex_rows],
        dtype=np.float64,
    ).reshape(-1, 3)
    tetrahedra = np.asarray(
        [[int(value) - 1 for value in row[:4]] for row in tetrahedron_rows],
        dtype=np.int64,
    ).reshape(-1, 4)
    tetrahedron_references = np.asarray(
        [int(row[4]) for row in tetrahedron_rows], dtype=np.int64
    )
    triangles = np.asarray(
        [[int(value) - 1 for value in row[:3]] for row in triangle_rows],
        dtype=np.int64,
    ).reshape(-1, 3)
    triangle_references = np.asarray(
        [int(row[3]) for row in triangle_rows], dtype=np.int64
    )
    if not np.isfinite(points).all():
        raise FloatingPointError("MMG3D returned non-finite coordinates")
    if len(points) == 0 or len(tetrahedra) == 0:
        raise RuntimeError("MMG3D returned an empty tetrahedral mesh")
    if tetrahedra.min() < 0 or tetrahedra.max() >= len(points):
        raise RuntimeError("MMG3D returned invalid tetrahedral connectivity")
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(points)):
        raise RuntimeError("MMG3D returned invalid triangular connectivity")
    return {
        "points": points,
        "tetrahedra": tetrahedra,
        "tetrahedron_references": tetrahedron_references,
        "triangles": triangles,
        "triangle_references": triangle_references,
    }


def run_isolated_mmg3d(
    command: list[str], working_directory: Path, timeout: float, label: str
) -> dict[str, Any]:
    """Run MMG3D with bounded diagnostics and a hard timeout."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=working_directory,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        timeout_stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        timeout_stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr
        )
        raise RuntimeError(
            f"{label} timed out after {duration:.1f} s "
            f"(limit {timeout:g} s); "
            f"stdout={_bounded_process_text(timeout_stdout)!r}; "
            f"stderr={_bounded_process_text(timeout_stderr)!r}"
        ) from None
    duration = time.monotonic() - started
    stdout = _bounded_process_text(completed.stdout)
    stderr = _bounded_process_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} {_returncode_description(completed.returncode)} after "
            f"{duration:.1f} s; stdout={stdout!r}; stderr={stderr!r}"
        )
    return {
        "command": command,
        "wall_time_seconds": float(duration),
        "returncode": int(completed.returncode),
        "stdout": stdout,
        "stderr": stderr,
    }


def compact_retained_levelset_mesh(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    cap_triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discard unused points and remap retained tets and cap triangles."""
    used = np.unique(np.asarray(tetrahedra, dtype=np.int64).ravel())
    mapping = np.full(len(points), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used), dtype=np.int64)
    compact_tetrahedra = mapping[np.asarray(tetrahedra, dtype=np.int64)]
    cap_triangles = np.asarray(cap_triangles, dtype=np.int64).reshape(-1, 3)
    if len(cap_triangles):
        retained_cap = np.all(mapping[cap_triangles] >= 0, axis=1)
        compact_cap = mapping[cap_triangles[retained_cap]]
    else:
        compact_cap = np.empty((0, 3), dtype=np.int64)
    return (
        np.ascontiguousarray(np.asarray(points, dtype=np.float64)[used]),
        np.ascontiguousarray(compact_tetrahedra),
        np.ascontiguousarray(compact_cap),
    )


def material_interface_triangles(
    tetrahedra: np.ndarray,
    tetrahedron_references: np.ndarray,
    first_reference: int,
    second_reference: int,
) -> np.ndarray:
    """Return faces shared by the two requested tetrahedral materials.

    MMG's material references are the robust description of the inserted
    level-set interface.  Coordinates alone are not sufficient immediately
    after remeshing because optimization may move an interface vertex a small
    distance away from the analytic zero set.
    """
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    references = np.asarray(tetrahedron_references, dtype=np.int64).reshape(-1)
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("Tetrahedral connectivity must have shape (N, 4)")
    if len(references) != len(tetrahedra):
        raise ValueError("Tetrahedral references do not match connectivity")

    local_faces = np.asarray(
        ([1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]), dtype=np.int64
    )
    faces = tetrahedra[:, local_faces].reshape(-1, 3)
    owners = np.repeat(np.arange(len(tetrahedra), dtype=np.int64), 4)
    keys = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise RuntimeError("MMG3D output contains a non-manifold tetrahedral face")

    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64))
    )
    paired_groups = np.flatnonzero(counts == 2)
    if len(paired_groups) == 0:
        return np.empty((0, 3), dtype=np.int64)
    first_face = order[starts[paired_groups]]
    second_face = order[starts[paired_groups] + 1]
    first_owner = owners[first_face]
    second_owner = owners[second_face]
    first_material = references[first_owner]
    second_material = references[second_owner]
    requested_pair = np.logical_or(
        np.logical_and(
            first_material == int(first_reference),
            second_material == int(second_reference),
        ),
        np.logical_and(
            first_material == int(second_reference),
            second_material == int(first_reference),
        ),
    )
    return np.ascontiguousarray(faces[first_face[requested_pair]])


def project_point_array_vertices_to_plane(
    points: np.ndarray,
    vertex_ids: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project selected array vertices onto one plane without moving others."""
    projected = np.asarray(points, dtype=np.float64).copy()
    vertex_ids = np.unique(np.asarray(vertex_ids, dtype=np.int64).reshape(-1))
    if len(vertex_ids) < 3:
        raise RuntimeError("Level-set interface contains fewer than three vertices")
    if vertex_ids.min() < 0 or vertex_ids.max() >= len(projected):
        raise ValueError("Level-set interface contains invalid vertex indices")
    unit_normal = np.asarray(normal, dtype=np.float64).reshape(3)
    residual = coordinates_along_normal(
        projected[vertex_ids], unit_normal
    ) - float(offset)
    before = float(np.max(np.abs(residual)))
    projected[vertex_ids] -= residual[:, None] * unit_normal[None, :]
    if not np.isfinite(projected).all():
        raise FloatingPointError("Exact interface projection produced non-finite points")
    after = float(
        np.max(
            np.abs(
                coordinates_along_normal(projected[vertex_ids], unit_normal)
                - float(offset)
            )
        )
    )
    return projected, {
        "method": "two_material_interface_normal_projection",
        "projected_vertices": int(len(vertex_ids)),
        "maximum_residual_before_projection": before,
        "maximum_residual_after_projection": after,
    }


def run_mmg3d_levelset_mode(
    shot_grid: vtk.vtkUnstructuredGrid,
    constraints: dict[str, Any],
    normal: np.ndarray,
    offset: float,
    retained_sign: int,
    executable: str,
    mode: str,
    args: argparse.Namespace,
) -> tuple[vtk.vtkUnstructuredGrid, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Insert an affine isosurface into the valid uncut shot volume."""
    with tempfile.TemporaryDirectory(prefix="ct55_mmg3d_levelset_") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "levelset_input.mesh"
        solution_path = temporary_path / "levelset_input.sol"
        parameter_path = temporary_path / "levelset_input.mmg3d"
        output_path = temporary_path / "levelset_output.mesh"
        write_medit_mesh(input_path, constraints)
        write_medit_scalar_solution(solution_path, constraints["phi"])
        write_mmg3d_levelset_parameters(parameter_path)
        local_edge = float(
            constraints["report"]["local_median_edge_length_mm"]
        )
        command = [
            executable,
            str(input_path),
            "-sol",
            str(solution_path),
            "-ls",
            "0",
            "-optim",
            "-nr",
            "-isoref",
            str(CAP_REFERENCE),
            "-out",
            str(output_path),
            "-hmin",
            f"{constraints['report']['mmg_hmin_mm']:.17g}",
            "-hmax",
            f"{constraints['report']['mmg_hmax_mm']:.17g}",
            "-hausd",
            f"{args.mmg_hausdorff:.17g}",
            "-hgrad",
            f"{args.mmg_gradation:.17g}",
            "-v",
            "-1",
        ]
        if mode == "nofem_fallback":
            command.append("-nofem")
        elif mode != "standard":
            raise ValueError(f"Unknown MMG3D level-set mode: {mode}")
        process_report = run_isolated_mmg3d(
            command,
            temporary_path,
            float(args.mmg_timeout_seconds),
            f"MMG3D level-set {mode}",
        )
        output = read_medit_mesh_with_references(output_path)

    points = np.asarray(output["points"], dtype=np.float64)
    tetrahedra = np.asarray(output["tetrahedra"], dtype=np.int64)
    tetrahedron_references = np.asarray(
        output["tetrahedron_references"], dtype=np.int64
    )
    triangles = np.asarray(output["triangles"], dtype=np.int64)
    triangle_references = np.asarray(
        output["triangle_references"], dtype=np.int64
    )
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    tolerance = max(diagonal * 1e-8, float(args.flat_tolerance) * 10.0, 1e-8)
    unique_references = np.unique(tetrahedron_references)
    allowed_references = {
        int(LEVELSET_DISCARD_REFERENCE),
        int(LEVELSET_RETAINED_REFERENCE),
    }
    unexpected_references = sorted(
        int(value)
        for value in unique_references
        if int(value) not in allowed_references
    )
    if unexpected_references:
        raise RuntimeError(
            "MMG3D returned unexpected tetrahedral material references: "
            + ", ".join(str(value) for value in unexpected_references)
        )
    retained = tetrahedron_references == LEVELSET_RETAINED_REFERENCE
    discarded = tetrahedron_references == LEVELSET_DISCARD_REFERENCE
    if not np.any(retained) or not np.any(discarded):
        raise RuntimeError(
            "MMG3D did not create both requested level-set material domains"
        )

    # The material interface is topologically authoritative.  MMG may move its
    # newly inserted vertices during -optim, so testing the analytic plane
    # before restoring those vertices causes false 'crossing tetrahedra'
    # failures.  Recover the shared interface, project only those vertices to
    # the exact plane, and then validate both material domains.
    cap_triangles = material_interface_triangles(
        tetrahedra,
        tetrahedron_references,
        LEVELSET_DISCARD_REFERENCE,
        LEVELSET_RETAINED_REFERENCE,
    )
    if len(cap_triangles) == 0:
        raise RuntimeError(
            "MMG3D created both materials but no conforming interface between them"
        )
    referenced_cap_triangles = triangles[triangle_references == CAP_REFERENCE]
    referenced_cap_keys = {
        tuple(int(value) for value in row)
        for row in np.sort(referenced_cap_triangles, axis=1)
    }
    material_cap_keys = {
        tuple(int(value) for value in row)
        for row in np.sort(cap_triangles, axis=1)
    }
    referenced_interface_triangles = len(
        referenced_cap_keys.intersection(material_cap_keys)
    )
    points, cap_projection = project_point_array_vertices_to_plane(
        points, cap_triangles.ravel(), normal, offset
    )
    depth = int(retained_sign) * (
        coordinates_along_normal(points, normal) - float(offset)
    )
    tet_depth = depth[tetrahedra]
    retained_minimum = np.min(tet_depth[retained], axis=1)
    discarded_maximum = np.max(tet_depth[discarded], axis=1)
    retained_side_failures = retained_minimum < -tolerance
    discarded_side_failures = discarded_maximum > tolerance
    if np.any(retained_side_failures) or np.any(discarded_side_failures):
        retained_penetration = max(0.0, -float(np.min(retained_minimum)))
        discarded_penetration = max(0.0, float(np.max(discarded_maximum)))
        raise RuntimeError(
            "MMG3D material domains remain inconsistent after exact interface "
            f"projection (retained failures={int(np.count_nonzero(retained_side_failures))}, "
            f"discarded failures={int(np.count_nonzero(discarded_side_failures))}, "
            f"maximum penetrations={retained_penetration:.6g}/"
            f"{discarded_penetration:.6g} mm)"
        )
    strict_crossing = np.logical_and(
        np.min(tet_depth, axis=1) < -tolerance,
        np.max(tet_depth, axis=1) > tolerance,
    )
    if np.any(strict_crossing):
        raise RuntimeError(
            "MMG3D output contains tetrahedra crossing the exact projected "
            f"interface ({int(np.count_nonzero(strict_crossing))} tetrahedra)"
        )
    reference_mapping_ok = True

    compact_points, compact_tetrahedra, compact_cap = (
        compact_retained_levelset_mesh(
            points, tetrahedra[retained], cap_triangles
        )
    )
    if len(compact_cap) == 0:
        raise RuntimeError("The retained level-set domain has no material-interface cap")
    grid = build_tetrahedral_grid(compact_points, compact_tetrahedra)
    final_projection = project_referenced_cap(grid, compact_cap.ravel(), normal, offset)
    cap_projection["final_compacted_projection"] = final_projection
    retained_points = np.asarray(
        vtk_to_numpy(grid.GetPoints().GetData()), dtype=np.float64
    )
    retained_depth = int(retained_sign) * (
        coordinates_along_normal(retained_points, normal) - float(offset)
    )
    maximum_removed_side_violation = max(0.0, -float(np.min(retained_depth)))
    if maximum_removed_side_violation > tolerance:
        raise RuntimeError("Retained level-set mesh contains removed-side points")

    frozen_report = verify_required_tetrahedra_preserved(
        shot_grid,
        grid,
        constraints["required_tetrahedra"],
        constraints["required_vertices"],
        constraints["boundary_faces"][constraints["required_triangles"]],
    )
    report = {
        "backend": "MMG3D_affine_levelset_on_valid_shot_volume",
        "executable": executable,
        "mode": mode,
        **process_report,
        "input_points": int(shot_grid.GetNumberOfPoints()),
        "input_tetrahedra": int(shot_grid.GetNumberOfCells()),
        "output_all_points": int(len(points)),
        "output_all_tetrahedra": int(len(tetrahedra)),
        "retained_points": int(grid.GetNumberOfPoints()),
        "retained_tetrahedra": int(grid.GetNumberOfCells()),
        "discarded_tetrahedra": int(np.count_nonzero(discarded)),
        "inserted_cap_triangles": int(len(compact_cap)),
        "material_interface_triangles": int(len(cap_triangles)),
        "referenced_cap_triangles": int(len(referenced_cap_triangles)),
        "referenced_material_interface_triangles": int(
            referenced_interface_triangles
        ),
        "cap_reference_coverage": float(
            referenced_interface_triangles / max(len(material_cap_keys), 1)
        ),
        "levelset_tolerance": float(tolerance),
        "maximum_removed_side_violation": float(maximum_removed_side_violation),
        "reference_mapping_ok": reference_mapping_ok,
        "tetrahedron_reference_counts": {
            str(int(reference)): int(np.count_nonzero(tetrahedron_references == reference))
            for reference in np.unique(tetrahedron_references)
        },
        "cap_reference": int(CAP_REFERENCE),
        "retained_reference": int(LEVELSET_RETAINED_REFERENCE),
        "discarded_reference": int(LEVELSET_DISCARD_REFERENCE),
        "local_median_edge_length_mm": local_edge,
        "hausdorff_tolerance_mm": float(args.mmg_hausdorff),
        "gradation": float(args.mmg_gradation),
        "ridge_detection_disabled_for_affine_levelset": True,
        "optimization_enabled_during_levelset_insertion": True,
    }
    return grid, report, cap_projection, frozen_report


def post_levelset_collar_constraints(
    grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    offset: float,
    retained_sign: int,
    wall_release_width: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Freeze deep anatomy but leave the cap and nearby wall jointly editable."""
    points = np.asarray(vtk_to_numpy(grid.GetPoints().GetData()), dtype=np.float64)
    tetrahedra = positively_oriented_tetrahedra(
        points, tetra_connectivity(grid)
    )
    topology = tetrahedral_face_topology(tetrahedra)
    boundary_faces = topology["boundary_faces"]
    boundary_owners = topology["boundary_owners"]
    adjacency = topology["adjacency"]
    depth = int(retained_sign) * (
        coordinates_along_normal(points, normal) - float(offset)
    )
    diagonal = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    tolerance = max(diagonal * 1e-8, float(args.flat_tolerance) * 10.0, 1e-8)
    if float(np.min(depth)) < -tolerance:
        raise RuntimeError("Level-set retained domain extends beyond its plane")
    residual = np.abs(
        coordinates_along_normal(points, normal) - float(offset)
    )
    cap_face = np.all(residual[boundary_faces] <= tolerance, axis=1)
    if not np.any(cap_face):
        raise RuntimeError("Level-set retained domain has no planar cap triangles")

    tetra_depth = np.min(depth[tetrahedra], axis=1)
    editable = tetra_depth <= float(args.collar_width) + tolerance
    editable[boundary_owners[cap_face]] = True
    editable = grow_tetrahedron_selection(
        editable, adjacency, int(args.collar_rings)
    )
    frozen = ~editable
    if not np.any(frozen):
        raise RuntimeError("Post-level-set collar covers the whole retained heart")

    boundary_minimum_depth = np.min(depth[boundary_faces], axis=1)
    required_triangle_mask = np.logical_and(
        ~cap_face, boundary_minimum_depth > float(wall_release_width)
    )
    required_triangles = np.flatnonzero(required_triangle_mask).astype(np.int64)
    required_tetrahedra = np.flatnonzero(frozen).astype(np.int64)
    required_vertices = np.unique(
        np.concatenate(
            (
                boundary_faces[required_triangles].ravel(),
                tetrahedra[required_tetrahedra].ravel(),
            )
        )
    ).astype(np.int64)
    triangle_references = np.where(
        cap_face, CAP_REFERENCE, ANATOMICAL_WALL_REFERENCE
    ).astype(np.int64)
    return {
        "points": points,
        "tetrahedra": tetrahedra,
        "boundary_faces": boundary_faces,
        "triangle_references": triangle_references,
        "cap_face": cap_face,
        "required_vertices": required_vertices,
        "required_triangles": required_triangles,
        "required_tetrahedra": required_tetrahedra,
        "frozen_tetrahedron_mask": frozen,
        "report": {
            "method": "post_levelset_editable_cap_wall_collar",
            "collar_width_mm": float(args.collar_width),
            "adjacency_rings": int(args.collar_rings),
            "wall_release_width_mm": float(wall_release_width),
            "editable_tetrahedra": int(np.count_nonzero(editable)),
            "frozen_tetrahedra": int(np.count_nonzero(frozen)),
            "required_nonbasal_triangles": int(len(required_triangles)),
            "required_vertices": int(len(required_vertices)),
            "cap_triangles_before_polish": int(np.count_nonzero(cap_face)),
            "cap_classification_tolerance": float(tolerance),
        },
    }


def evaluate_levelset_output(
    grid: vtk.vtkUnstructuredGrid,
    reference_surface: vtk.vtkPolyData,
    reference_surface_quality: dict[str, Any],
    reference_cap_topology: dict[str, Any],
    raw_surface: np.ndarray,
    template_faces: np.ndarray,
    template_object_labels: np.ndarray,
    shot_grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    offset: float,
    candidate: dict[str, Any],
    levelset_report: dict[str, Any],
    cap_projection: dict[str, Any],
    frozen_report: dict[str, Any],
    protection_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Perform complete post-level-set geometry and simulation-mesh QA."""
    metadata = transfer_volume_metadata(shot_grid, grid)
    final_surface, basal_face, basal_vertex = boundary_from_volume(
        grid, normal, offset
    )
    anatomical_labels = add_anatomical_object_labels(
        final_surface,
        raw_surface,
        template_faces,
        template_object_labels,
        basal_face,
    )
    volume_quality = clipped_volume_quality(grid)
    surface_quality = extracted_surface_quality(
        final_surface, basal_face, basal_vertex, normal, offset
    )
    final_points, final_faces = polydata_points_and_faces(final_surface)
    final_cap_topology = basal_cap_topology(final_faces, basal_face)
    fidelity = bidirectional_surface_fidelity(reference_surface, final_surface)
    diagonal = max(float(np.linalg.norm(np.ptp(final_points, axis=0))), 1.0)
    local_edge = float(levelset_report["local_median_edge_length_mm"])
    fidelity_tolerance = max(
        diagonal * 1e-6,
        float(args.mmg_hausdorff) * 4.0,
        local_edge * 0.25,
        1e-5,
    )
    fidelity_p99_tolerance = max(
        diagonal * 5e-7,
        float(args.mmg_hausdorff) * 3.0,
        local_edge * 0.15,
        1e-5,
    )
    fidelity_rms_tolerance = max(
        diagonal * 1e-7,
        float(args.mmg_hausdorff) * 2.0,
        local_edge * 0.05,
        1e-6,
    )
    # Zero identifies the artificial basal closure.  Only the positive IDs
    # are anatomical objects (outer wall, LV cavity and RV cavity).
    expected_objects = sorted(
        int(value)
        for value in np.unique(template_object_labels)
        if int(value) > 0
    )
    output_anatomical_objects = sorted(
        int(value)
        for value in anatomical_labels["output_objects"]
        if int(value) > 0
    )
    checks = {
        "positive_tetrahedra": volume_quality["nonpositive_scaled_jacobians"] == 0,
        "finite_tetrahedral_quality": volume_quality[
            "nonfinite_scaled_jacobians"
        ]
        == 0,
        "nondegenerate_tetrahedra": volume_quality["degenerate_tetrahedra"] == 0,
        "minimum_scaled_jacobian": volume_quality["minimum_scaled_jacobian"]
        is not None
        and volume_quality["minimum_scaled_jacobian"]
        >= args.minimum_scaled_jacobian,
        "p01_scaled_jacobian": volume_quality["p01_scaled_jacobian"] is not None
        and volume_quality["p01_scaled_jacobian"]
        >= args.minimum_p01_scaled_jacobian,
        "minimum_dihedral_angle": volume_quality["minimum_dihedral_angle"]
        is not None
        and volume_quality["minimum_dihedral_angle"]
        >= args.minimum_dihedral_angle,
        "flat_base": surface_quality["maximum_basal_plane_error"]
        <= args.flat_tolerance,
        "closed_surface": surface_quality["boundary_edges"] == 0,
        "manifold_surface": surface_quality["nonmanifold_edges"] == 0,
        "nondegenerate_surface": surface_quality["degenerate_triangles"] == 0,
        "expected_outer_lv_rv_contours": (
            args.expected_basal_contours == 0
            or final_cap_topology["contour_loops"] == args.expected_basal_contours
        ),
        "reference_chop_had_expected_contours": (
            args.expected_basal_contours == 0
            or reference_cap_topology["contour_loops"]
            == args.expected_basal_contours
        ),
        # Only the basal collar is remeshed.  Protected wall geometry is
        # checked exactly below; within the collar, use scale-aware maximum,
        # p99 and RMS bounds so different planar-cap triangulations are not
        # mistaken for anatomical distortion.
        "surface_matches_reference_chop": (
            fidelity["maximum_bidirectional_distance"] <= fidelity_tolerance
            and fidelity["p99_bidirectional_distance"]
            <= fidelity_p99_tolerance
            and fidelity["rms_bidirectional_distance"]
            <= fidelity_rms_tolerance
        ),
        "all_anatomical_objects_preserved": output_anatomical_objects
        == expected_objects,
        "basal_cap_has_dedicated_zero_label": (
            0 not in anatomical_labels["source_objects"]
            or anatomical_labels["basal_triangles_assigned_zero"]
            == anatomical_labels["basal_triangles"]
        ),
        "protected_anatomy_preserved": bool(frozen_report["passed"]),
        "levelset_reference_mapping": bool(
            levelset_report.get("reference_mapping_ok", False)
        ),
        "retained_side_only": float(
            levelset_report.get("maximum_removed_side_violation", math.inf)
        )
        <= max(args.flat_tolerance, 1e-7),
    }
    return {
        "grid": grid,
        "surface": final_surface,
        "candidate": candidate,
        "levelset": levelset_report,
        "cap_projection": cap_projection,
        "frozen_region": frozen_report,
        "protection": protection_report,
        "volume_metadata": metadata,
        "anatomical_label_transfer": anatomical_labels,
        "reference_surface_quality": reference_surface_quality,
        "reference_cap_topology": reference_cap_topology,
        "volume_quality": volume_quality,
        "surface_quality": surface_quality,
        "final_cap_topology": final_cap_topology,
        "surface_fidelity": fidelity,
        "surface_fidelity_tolerance": fidelity_tolerance,
        "surface_fidelity_p99_tolerance": fidelity_p99_tolerance,
        "surface_fidelity_rms_tolerance": fidelity_rms_tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def levelset_bundle_score(bundle: dict[str, Any]) -> tuple[float, ...]:
    quality = bundle["volume_quality"]
    return (
        float(bool(bundle["passed"])),
        float(quality["minimum_scaled_jacobian"] or -1.0),
        float(quality["p01_scaled_jacobian"] or -1.0),
        float(quality["minimum_dihedral_angle"] or -1.0),
        -float(bundle["candidate"]["extra_inward_depth"]),
    )


def polish_levelset_bundle(
    initial: dict[str, Any],
    original_constraints: dict[str, Any],
    reference_surface: vtk.vtkPolyData,
    reference_surface_quality: dict[str, Any],
    reference_cap_topology: dict[str, Any],
    raw_surface: np.ndarray,
    template_faces: np.ndarray,
    object_labels: np.ndarray,
    shot_grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    retained_sign: int,
    executable: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Apply optional local MMG quality passes to a valid level-set result."""
    best = initial
    current = initial
    history: list[dict[str, Any]] = []
    if current["passed"] or args.mmg_quality_passes == 0:
        current["quality_polish_attempts"] = history
        return current
    offset = float(initial["candidate"]["offset"])
    wall_release = float(
        original_constraints["report"]["wall_release_width_mm"]
    )
    for pass_index in range(1, int(args.mmg_quality_passes) + 1):
        constraints = post_levelset_collar_constraints(
            current["grid"],
            normal,
            offset,
            retained_sign,
            wall_release,
            args,
        )
        modes = ["standard"]
        if args.mmg_nofem_fallback:
            modes.append("nofem_fallback")
        pass_bundles: list[dict[str, Any]] = []
        for mode in modes:
            attempt: dict[str, Any] = {
                "pass": pass_index,
                "mode": mode,
                "collar": constraints["report"],
            }
            try:
                grid, mmg_report, cap_projection, _ = run_mmg3d_collar_mode(
                    current["grid"],
                    constraints,
                    normal,
                    offset,
                    executable,
                    mode,
                    args,
                )
                cumulative_frozen = verify_required_tetrahedra_preserved(
                    shot_grid,
                    grid,
                    original_constraints["required_tetrahedra"],
                    original_constraints["required_vertices"],
                    original_constraints["boundary_faces"][
                        original_constraints["required_triangles"]
                    ],
                )
                levelset_report = dict(initial["levelset"])
                levelset_report["post_levelset_quality_passes"] = pass_index
                bundle = evaluate_levelset_output(
                    grid,
                    reference_surface,
                    reference_surface_quality,
                    reference_cap_topology,
                    raw_surface,
                    template_faces,
                    object_labels,
                    shot_grid,
                    normal,
                    offset,
                    initial["candidate"],
                    levelset_report,
                    cap_projection,
                    cumulative_frozen,
                    original_constraints["report"],
                    args,
                )
                bundle["levelset_insertion"] = initial["levelset_insertion"]
                bundle["quality_polish"] = mmg_report
                bundle["quality_polish_collar"] = constraints["report"]
                pass_bundles.append(bundle)
                attempt.update(
                    {
                        "passed": bool(bundle["passed"]),
                        "checks": bundle["checks"],
                        "mmg3d": mmg_report,
                        "volume_quality": bundle["volume_quality"],
                        "frozen_region": cumulative_frozen,
                    }
                )
            except Exception as exc:
                attempt.update({"passed": False, "error": str(exc)})
            history.append(attempt)
        if not pass_bundles:
            break
        current = max(pass_bundles, key=levelset_bundle_score)
        if levelset_bundle_score(current) > levelset_bundle_score(best):
            best = current
        if current["passed"]:
            break
    best["quality_polish_attempts"] = history
    return best


def remesh_one_levelset_candidate(
    raw_surface: np.ndarray,
    template_faces: np.ndarray,
    object_labels: np.ndarray,
    shot_grid: vtk.vtkUnstructuredGrid,
    normal: np.ndarray,
    retained_sign: int,
    candidate: dict[str, Any],
    executable: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Insert one candidate plane into the valid shot mesh and perform QA."""
    offset = float(candidate["offset"])
    reference_surface, _, _, reference_quality, reference_topology = (
        clip_and_cap_surface(
            raw_surface,
            template_faces,
            normal,
            offset,
            retained_sign,
            args.flat_tolerance,
        )
    )
    constraints = levelset_protection_constraints(
        shot_grid, normal, offset, retained_sign, args
    )
    modes = ["standard"]
    if args.mmg_nofem_fallback:
        modes.append("nofem_fallback")
    attempts: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    for mode in modes:
        try:
            grid, levelset_report, cap_projection, frozen_report = (
                run_mmg3d_levelset_mode(
                    shot_grid,
                    constraints,
                    normal,
                    offset,
                    retained_sign,
                    executable,
                    mode,
                    args,
                )
            )
            bundle = evaluate_levelset_output(
                grid,
                reference_surface,
                reference_quality,
                reference_topology,
                raw_surface,
                template_faces,
                object_labels,
                shot_grid,
                normal,
                offset,
                candidate,
                levelset_report,
                cap_projection,
                frozen_report,
                constraints["report"],
                args,
            )
            bundle["levelset_insertion"] = levelset_report
            bundle = polish_levelset_bundle(
                bundle,
                constraints,
                reference_surface,
                reference_quality,
                reference_topology,
                raw_surface,
                template_faces,
                object_labels,
                shot_grid,
                normal,
                retained_sign,
                executable,
                args,
            )
            bundles.append(bundle)
            attempts.append(
                {
                    "mode": mode,
                    "passed": bool(bundle["passed"]),
                    "checks": bundle["checks"],
                    "levelset": levelset_report,
                    "quality_polish_attempts": bundle[
                        "quality_polish_attempts"
                    ],
                    "volume_quality": bundle["volume_quality"],
                    "protected_anatomy": bundle["frozen_region"],
                }
            )
            if bundle["passed"]:
                bundle["levelset_mode_attempts"] = attempts
                return bundle
        except Exception as exc:
            attempts.append({"mode": mode, "passed": False, "error": str(exc)})
    if not bundles:
        reasons = " | ".join(
            f"{attempt['mode']}: {attempt.get('error', 'failed QA')}"
            for attempt in attempts
        )
        raise RuntimeError("MMG3D level-set insertion failed: " + reasons)
    best = max(bundles, key=levelset_bundle_score)
    best["levelset_mode_attempts"] = attempts
    return best


def quality_driven_mmg_levelset_remesh(
    raw_surface: np.ndarray,
    template: dict[str, Any],
    shot_grid: vtk.vtkUnstructuredGrid,
    shot_volume_points: np.ndarray,
    normal: np.ndarray,
    plane: dict[str, Any],
    executable: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select the shallowest topology-valid level-set cut passing all QA."""
    candidates = candidate_cut_offsets(
        shot_volume_points,
        plane,
        float(plane["ct55_offset"]),
        args.maximum_plane_offset_shift,
        args.cut_search_distance,
        args.cut_search_step,
        args.cut_search_maximum_candidates,
    )
    retained_sign = int(plane["retained_implicit_sign"])
    template_faces = np.asarray(template["faces"], dtype=np.int64)
    object_labels = np.asarray(template["object_labels"], dtype=np.int32)
    prescreen: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        record: dict[str, Any] = {
            "candidate": index,
            "offset": float(candidate["offset"]),
            "shift_from_ct55": float(candidate["offset"])
            - float(plane["ct55_offset"]),
            "extra_inward_depth": float(candidate["extra_inward_depth"]),
            "nearest_volume_vertex_distance": float(
                candidate["nearest_volume_vertex_distance"]
            ),
        }
        record["nearest_surface_vertex_distance"] = record[
            "nearest_volume_vertex_distance"
        ]
        try:
            _, basal_face, _, surface_quality, topology = clip_and_cap_surface(
                raw_surface,
                template_faces,
                normal,
                float(candidate["offset"]),
                retained_sign,
                args.flat_tolerance,
            )
            minimum_clearance = float(plane["cut_clearance"]) + float(
                candidate["extra_inward_depth"]
            )
            checks = {
                "complete_old_base_removed": bool(
                    plane["complete_cut_within_bound"]
                )
                and minimum_clearance >= args.cut_clearance - 1e-10,
                "flat_reference_cap": surface_quality[
                    "maximum_basal_plane_error"
                ]
                <= args.flat_tolerance,
                "closed_reference_surface": surface_quality["boundary_edges"] == 0,
                "manifold_reference_surface": surface_quality[
                    "nonmanifold_edges"
                ]
                == 0,
                "nondegenerate_reference_surface": surface_quality[
                    "degenerate_triangles"
                ]
                == 0,
                "retained_side": surface_quality[
                    "maximum_retained_side_violation"
                ]
                <= max(args.flat_tolerance, 1e-8),
                "cap_has_triangles": int(np.count_nonzero(basal_face)) > 0,
                "cap_contours_are_loops": topology[
                    "every_boundary_vertex_has_degree_two"
                ],
                "expected_outer_lv_rv_contours": (
                    args.expected_basal_contours == 0
                    or topology["contour_loops"] == args.expected_basal_contours
                ),
            }
            record.update(
                {
                    "minimum_old_base_clearance": minimum_clearance,
                    "surface_quality": surface_quality,
                    "cap_topology": topology,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
            if record["passed"]:
                valid.append(record)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            record.update({"passed": False, "error": str(exc)})
        prescreen.append(record)

    valid.sort(key=lambda value: value["extra_inward_depth"])
    selected_records: list[dict[str, Any]] = []
    minimum_separation = max(float(args.cut_search_step) * 0.45, 1e-6)
    for record in valid:
        if (
            not selected_records
            or record["extra_inward_depth"]
            - selected_records[-1]["extra_inward_depth"]
            >= minimum_separation
        ):
            selected_records.append(record)
        elif (
            record["nearest_volume_vertex_distance"]
            > selected_records[-1]["nearest_volume_vertex_distance"]
        ):
            selected_records[-1] = record
        if len(selected_records) >= args.remesh_candidates:
            break

    attempts: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for remesh_index, record in enumerate(selected_records, start=1):
        print(
            f"  level-set candidate {remesh_index}/{len(selected_records)}: "
            f"cut depth={record['extra_inward_depth']:.4g} mm",
            flush=True,
        )
        attempt: dict[str, Any] = {
            "candidate": int(record["candidate"]),
            "offset": float(record["offset"]),
            "shift_from_ct55": float(record["shift_from_ct55"]),
            "extra_inward_depth": float(record["extra_inward_depth"]),
            "nearest_volume_vertex_distance": float(
                record["nearest_volume_vertex_distance"]
            ),
        }
        try:
            bundle = remesh_one_levelset_candidate(
                raw_surface,
                template_faces,
                object_labels,
                shot_grid,
                normal,
                retained_sign,
                record,
                executable,
                args,
            )
            bundles.append(bundle)
            attempt.update(
                {
                    "passed": bool(bundle["passed"]),
                    "checks": bundle["checks"],
                    "levelset": bundle["levelset"],
                    "levelset_mode_attempts": bundle[
                        "levelset_mode_attempts"
                    ],
                    "quality_polish_attempts": bundle[
                        "quality_polish_attempts"
                    ],
                    "protection": bundle["protection"],
                    "frozen_region": bundle["frozen_region"],
                    "volume_quality": bundle["volume_quality"],
                    "surface_quality": bundle["surface_quality"],
                    "cap_topology": bundle["final_cap_topology"],
                }
            )
            if bundle["passed"]:
                selected = bundle
                attempts.append(attempt)
                break
        except Exception as exc:
            attempt.update({"passed": False, "error": str(exc)})
            print(
                "    failed safely: " + _bounded_process_text(str(exc), 1200),
                flush=True,
            )
        attempts.append(attempt)
    if selected is None and bundles:
        selected = max(bundles, key=levelset_bundle_score)
    search = {
        "method": "MMG3D_affine_levelset_on_uncut_valid_shot_volume",
        "selection_rule": (
            "shallowest fully passing candidate; best-quality attempted mesh "
            "is returned but remains rejected when none passes"
        ),
        "candidate_planes_constructed": len(candidates),
        "surface_valid_candidates": len(valid),
        "levelset_candidates_requested": int(args.remesh_candidates),
        "levelset_candidates_attempted": len(attempts),
        "passing_levelset_meshes": int(
            sum(bool(item.get("passed")) for item in attempts)
        ),
        "passing_candidate_found": bool(
            selected is not None and selected["passed"]
        ),
        "surface_prescreen": prescreen,
        "levelset_attempts": attempts,
    }
    return selected, search


def _legacy_direct_clip_main_disabled() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    template = read_template_with_surface_fallback(
        args.template_prepared, args.template_surface, args.template_labeled_surface
    )
    volume = read_volume(args.template_volume)
    template_volume_points = np.asarray(
        vtk_to_numpy(volume.GetPoints().GetData()), dtype=np.float64
    )
    template_tetrahedra = tetra_connectivity(volume)
    surface_points = np.asarray(template["points"], dtype=np.float64)
    basal_ids = np.asarray(template["basal_ids"], dtype=np.int64)
    normal = np.asarray(template["plane_normal"], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= 1e-12:
        raise ValueError("CT55 basal-plane normal is invalid")
    ct55_offset = float(template["plane_offset"]) / normal_norm
    normal = normal / normal_norm

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. Use another "
            "directory or pass --resume after an interrupted run."
        )
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    samples, sampling_diagnostics = decode_and_save_momenta(
        args, template, args.output_dir
    )
    surfaces_dir = args.output_dir / "GeneratedSurfaces"
    volumes_dir = args.output_dir / "GeneratedVolumes"
    reports_dir = args.output_dir / "Reports"
    momenta_dir = args.output_dir / "GeneratedMomentas"
    for directory in (surfaces_dir, volumes_dir, reports_dir, momenta_dir):
        directory.mkdir(exist_ok=args.resume)

    print(f"Samples:              {len(samples)}")
    print(f"Latent sampling:      {args.sampling_policy}")
    print(f"Latent temperature:   {args.temperature:g}")
    print(f"CT55 volume:          {volume.GetNumberOfPoints():,} points; "
          f"{volume.GetNumberOfCells():,} tetrahedra")
    print("Basal method:         shoot volume, then complete inward plane clip")
    print(f"Maximum cut shift:    {args.maximum_plane_offset_shift:g} mm")
    print(f"Cut clearance:        {args.cut_clearance:g} mm")
    print(
        f"Quality plane search: 0–{args.cut_search_distance:g} mm inward; "
        f"nominal step {args.cut_search_step:g} mm"
    )
    print(
        f"Required min quality: scaled Jacobian >= "
        f"{args.minimum_scaled_jacobian:g}"
    )
    print("Momentum refinement:  none\n")

    rows: list[dict[str, Any]] = []
    for sequence, (sample_index, sample) in enumerate(samples, start=1):
        sample_id = f"generated_{sample_index:04d}"
        print(f"[{sequence}/{len(samples)}] {sample_id}", flush=True)
        momentum_path = momenta_dir / f"{sample_id}_momenta.npz"
        surface_path = surfaces_dir / f"{sample_id}.vtk"
        volume_path = volumes_dir / f"{sample_id}.vtu"
        report_path = reports_dir / f"{sample_id}.json"
        expected = (momentum_path, surface_path, volume_path, report_path)
        if report_path.exists() and args.resume:
            if not all(path.is_file() for path in expected):
                raise FileNotFoundError(f"Incomplete resumed sample: {sample_id}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows.append(report["summary_row"])
            print(f"  resumed: preserving {report['status']}")
            continue
        if any(path.exists() for path in expected):
            raise FileExistsError(f"Partial or existing output for {sample_id}")

        raw_surface = shoot_numpy_points(
            sample["control_points"], sample["momenta"], surface_points,
            sample["origin"], sample["scale"], sample["steps"],
            args.shooting_chunk_size, args.threads,
        )
        shot_volume_points = shoot_numpy_points(
            sample["control_points"], sample["momenta"], template_volume_points,
            sample["origin"], sample["scale"], sample["steps"],
            args.shooting_chunk_size, args.threads,
        )
        shot_quality = tetrahedron_qa(
            template_volume_points, shot_volume_points, template_tetrahedra
        )
        plane = choose_complete_cut_plane(
            raw_surface, basal_ids, normal, ct55_offset,
            args.maximum_plane_offset_shift, args.cut_clearance,
        )
        shot_grid = deformed_grid(volume, shot_volume_points)
        shot_absolute_quality = clipped_volume_quality(shot_grid)
        (
            clipped,
            boundary,
            basal_face,
            basal_vertex,
            volume_quality,
            surface_quality,
            plane,
            cut_search,
        ) = quality_aware_clip_search(
            shot_grid, shot_volume_points, normal, plane, args
        )
        print(
            f"  cut search tested {cut_search['candidates_tested']} candidate(s); "
            f"extra inward depth={plane['quality_search_extra_inward_depth']:.4g} mm; "
            f"passing candidate={'yes' if cut_search['passing_candidate_found'] else 'no'}",
            flush=True,
        )

        np.savez_compressed(
            momentum_path,
            control_points=np.asarray(sample["control_points"], dtype=np.float64),
            momenta=np.asarray(sample["momenta"], dtype=np.float64),
            origin=np.asarray(sample["origin"], dtype=np.float64),
            scale=np.float64(sample["scale"]),
            steps=np.int64(sample["steps"]),
            plane_normal=normal,
            plane_offset=np.float64(plane["selected_offset"]),
            ct55_plane_offset=np.float64(ct55_offset),
            template="ct_case_0055",
            target=sample_id,
            latent=np.asarray(sample["latent"], dtype=np.float32),
            temperature=np.float32(sample["temperature"]),
            sampling_policy=np.asarray(sample["sampling_policy"]),
            posterior_component_index=np.int64(
                -1 if sample["posterior_component_index"] is None
                else sample["posterior_component_index"]
            ),
            clipping_version=np.asarray(CLIPPING_VERSION),
        )
        write_unstructured_grid(volume_path, clipped)
        write_polydata(surface_path, boundary)

        checks = {
            "complete_cut_within_bound": plane["complete_cut_within_bound"],
            "cut_clearance_achieved": plane["minimum_achieved_clearance"]
            >= args.cut_clearance - 1e-7,
            "flat_base": surface_quality["maximum_basal_plane_error"]
            <= args.flat_tolerance,
            "closed_surface": surface_quality["boundary_edges"] == 0,
            "surface_manifold": surface_quality["nonmanifold_edges"] == 0,
            "surface_nondegenerate": surface_quality["degenerate_triangles"] == 0,
            "raw_shoot_not_inverted": shot_quality["inverted_tetrahedra"] == 0,
            "clipped_tetrahedra_nondegenerate": volume_quality["degenerate_tetrahedra"] == 0,
            "clipped_tetrahedra_positive_quality":
            volume_quality["nonpositive_scaled_jacobians"] == 0,
            "clipped_tetrahedra_quality":
            volume_quality["minimum_scaled_jacobian"] is not None
            and volume_quality["minimum_scaled_jacobian"]
            >= args.minimum_scaled_jacobian,
            "quality_aware_cut_found": cut_search["passing_candidate_found"],
        }
        status = "ACCEPTED" if all(checks.values()) else "REJECTED"
        row = {
            "sample": sample_id,
            "status": status,
            "plane_offset_shift": plane["selected_shift_from_ct55"],
            "plane_offset_clamped": plane["was_clamped"],
            "cut_extra_inward_depth": plane[
                "quality_search_extra_inward_depth"
            ],
            "cut_candidates_tested": cut_search["candidates_tested"],
            "flatness_error": surface_quality["maximum_basal_plane_error"],
            "minimum_scaled_jacobian": volume_quality["minimum_scaled_jacobian"],
            "clipped_tetrahedra": volume_quality["tetrahedra"],
            "surface_triangles": surface_quality["triangles"],
            "surface": str(surface_path),
            "volume": str(volume_path),
            "momenta": str(momentum_path),
        }
        report = {
            "status": status,
            "sample": sample_id,
            "generation_version": GENERATION_VERSION,
            "clipping_version": CLIPPING_VERSION,
            "template": "ct_case_0055",
            "scale_performed": False,
            "momentum_refinement_performed": False,
            "volume_clip_performed": True,
            "surface_clip_performed": False,
            "plane_selection": plane,
            "cut_search": cut_search,
            "shot_volume_quality": shot_quality,
            "shot_volume_absolute_quality": shot_absolute_quality,
            "clipped_volume_quality": volume_quality,
            "extracted_surface_quality": surface_quality,
            "checks": checks,
            "outputs": {
                "generated_momenta": str(momentum_path),
                "clipped_tetrahedral_vtu": str(volume_path),
                "volume_derived_surface_vtk": str(surface_path),
            },
            "summary_row": row,
        }
        write_json(report_path, report)
        rows.append(row)
        print(
            f"  {status}: plane={surface_quality['maximum_basal_plane_error']:.3g} mm; "
            f"shift={plane['selected_shift_from_ct55']:+.3f} mm; "
            f"min scaled Jacobian={volume_quality['minimum_scaled_jacobian']:.4g}; "
            f"tets={volume_quality['tetrahedra']:,}",
            flush=True,
        )

    with (args.output_dir / "generation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "generation_complete",
        "processed": len(rows),
        "accepted": sum(row["status"] == "ACCEPTED" for row in rows),
        "rejected": sum(row["status"] == "REJECTED" for row in rows),
        "generation_version": GENERATION_VERSION,
        "clipping_version": CLIPPING_VERSION,
        "sampling": sampling_diagnostics,
        "template_volume": str(args.template_volume),
        "template_volume_sha256": sha256(args.template_volume),
        "rows": rows,
    }
    write_json(args.output_dir / "generation_summary.json", summary)
    print(
        f"\nGeneration complete: {summary['accepted']}/{summary['processed']} accepted."
    )
    return 0 if summary["rejected"] == 0 else 2


def levelset_generation_main() -> int:
    """Generate unchanged-momentum hearts with a protected MMG level-set cut."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    mmg_executable, mmg_probe = resolve_mmg3d_executable(args.mmg3d_executable)

    template = read_template_with_surface_fallback(
        args.template_prepared,
        args.template_surface,
        args.template_labeled_surface,
    )
    volume = read_volume(args.template_volume)
    template_volume_points = np.asarray(
        vtk_to_numpy(volume.GetPoints().GetData()), dtype=np.float64
    )
    template_tetrahedra = tetra_connectivity(volume)
    surface_points = np.asarray(template["points"], dtype=np.float64)
    template_faces = np.asarray(template["faces"], dtype=np.int64)
    basal_ids = np.asarray(template["basal_ids"], dtype=np.int64)
    _, template_surface_volume_error = surface_to_volume_mapping(
        surface_points, volume
    )
    normal = np.asarray(template["plane_normal"], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= 1e-12:
        raise ValueError("CT55 basal-plane normal is invalid")
    ct55_offset = float(template["plane_offset"]) / normal_norm
    normal = normal / normal_norm

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. Use a new "
            "directory, or use --resume only for the same interrupted run."
        )
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    surfaces_dir = args.output_dir / "GeneratedSurfaces"
    volumes_dir = args.output_dir / "GeneratedVolumes"
    reports_dir = args.output_dir / "Reports"
    momenta_dir = args.output_dir / "GeneratedMomentas"
    for directory in (surfaces_dir, volumes_dir, reports_dir, momenta_dir):
        directory.mkdir(exist_ok=args.resume)

    samples, sampling_diagnostics = decode_and_save_momenta(
        args, template, args.output_dir
    )
    print(f"Samples:                {len(samples)}")
    print(f"Latent sampling:        {args.sampling_policy}")
    print(f"Latent temperature:     {args.temperature:g}")
    print(
        f"CT55 volume:            {volume.GetNumberOfPoints():,} points; "
        f"{volume.GetNumberOfCells():,} tetrahedra"
    )
    print(
        "Basal method:           shoot -> MMG3D affine level-set cut -> "
        "protected local polish"
    )
    print(
        f"Plane search:           up to {args.cut_search_distance:g} mm inward; "
        f"{args.remesh_candidates} shallow-first level-set attempts"
    )
    print(
        f"Required tet quality:   min SJ {args.minimum_scaled_jacobian:g}; "
        f"p01 SJ {args.minimum_p01_scaled_jacobian:g}; "
        f"min angle {args.minimum_dihedral_angle:g} deg"
    )
    print(f"Required flatness:      {args.flat_tolerance:g} mm")
    print(
        f"Editable collar:        {args.collar_width:g} mm plus "
        f"{args.collar_rings} adjacency rings"
    )
    print(
        "Wall release:           "
        + (
            "edge-length-aware automatic band"
            if args.wall_release_width == 0
            else f"{args.wall_release_width:g} mm"
        )
    )
    print(f"Quality polish passes:  {args.mmg_quality_passes}")
    print("Collar relocation:      enabled for tetrahedral quality")
    print("Protected anatomy:      fixed non-basal boundary vertices/triangles")
    print(f"MMG3D executable:       {mmg_executable}")
    print(f"MMG3D isolation:        {args.mmg_timeout_seconds:g} s timeout per attempt")
    print("Momentum refinement:    none")
    print("Direct tet clipping:    none; MMG inserts a conforming zero isosurface")
    print("Anatomical smoothing:   none\n")

    rows: list[dict[str, Any]] = []
    for sequence, (sample_index, sample) in enumerate(samples, start=1):
        sample_id = f"generated_{sample_index:04d}"
        print(f"[{sequence}/{len(samples)}] {sample_id}", flush=True)
        momentum_path = momenta_dir / f"{sample_id}_momenta.npz"
        surface_path = surfaces_dir / f"{sample_id}.vtk"
        volume_path = volumes_dir / f"{sample_id}.vtu"
        report_path = reports_dir / f"{sample_id}.json"
        possible_outputs = (momentum_path, surface_path, volume_path, report_path)

        if report_path.is_file() and args.resume:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for saved_path in report.get("outputs", {}).values():
                if saved_path and not Path(saved_path).is_file():
                    raise FileNotFoundError(
                        f"Incomplete resumed sample {sample_id}: missing {saved_path}"
                    )
            rows.append(report["summary_row"])
            print(f"  resumed: preserving {report['status']}")
            continue
        if any(path.exists() for path in possible_outputs):
            raise FileExistsError(
                f"Partial output already exists for {sample_id}; do not mix runs"
            )

        np.savez_compressed(
            momentum_path,
            control_points=np.asarray(sample["control_points"], dtype=np.float64),
            momenta=np.asarray(sample["momenta"], dtype=np.float64),
            origin=np.asarray(sample["origin"], dtype=np.float64),
            scale=np.float64(sample["scale"]),
            steps=np.int64(sample["steps"]),
            plane_normal=normal,
            ct55_plane_offset=np.float64(ct55_offset),
            template=np.asarray("ct_case_0055"),
            target=np.asarray(sample_id),
            latent=np.asarray(sample["latent"], dtype=np.float32),
            temperature=np.float32(sample["temperature"]),
            sampling_policy=np.asarray(sample["sampling_policy"]),
            posterior_component_index=np.int64(
                -1
                if sample["posterior_component_index"] is None
                else sample["posterior_component_index"]
            ),
            posterior_component_source=np.asarray(
                ""
                if sample["posterior_component_source"] is None
                else sample["posterior_component_source"]
            ),
            generation_version=np.asarray(GENERATION_VERSION),
            clipping_version=np.asarray(CLIPPING_VERSION),
            momentum_refinement_performed=np.asarray(False),
        )

        status = "REJECTED"
        selected_plane: dict[str, Any] | None = None
        selected_bundle: dict[str, Any] | None = None
        cut_search: dict[str, Any] = {}
        error: str | None = None
        shot_quality: dict[str, Any] | None = None
        shot_absolute_quality: dict[str, Any] | None = None
        try:
            raw_surface = shoot_numpy_points(
                sample["control_points"],
                sample["momenta"],
                surface_points,
                sample["origin"],
                sample["scale"],
                sample["steps"],
                args.shooting_chunk_size,
                args.threads,
            )
            shot_volume_points = shoot_numpy_points(
                sample["control_points"],
                sample["momenta"],
                template_volume_points,
                sample["origin"],
                sample["scale"],
                sample["steps"],
                args.shooting_chunk_size,
                args.threads,
            )
            shot_quality = tetrahedron_qa(
                template_volume_points, shot_volume_points, template_tetrahedra
            )
            shot_grid = deformed_grid(volume, shot_volume_points)
            shot_absolute_quality = clipped_volume_quality(shot_grid)
            plane = choose_complete_cut_plane(
                raw_surface,
                basal_ids,
                normal,
                ct55_offset,
                args.maximum_plane_offset_shift,
                args.cut_clearance,
            )
            selected_bundle, cut_search = quality_driven_mmg_levelset_remesh(
                raw_surface,
                template,
                shot_grid,
                shot_volume_points,
                normal,
                plane,
                mmg_executable,
                args,
            )
            if selected_bundle is not None:
                candidate = selected_bundle["candidate"]
                selected_plane = dict(plane)
                selected_plane.update(
                    {
                        "selected_offset": float(candidate["offset"]),
                        "selected_shift_from_ct55": float(candidate["shift_from_ct55"]),
                        "quality_search_extra_inward_depth": float(
                            candidate["extra_inward_depth"]
                        ),
                        "nearest_volume_vertex_distance": float(
                            candidate["nearest_volume_vertex_distance"]
                        ),
                        "selection": "shallowest_passing_MMG3D_levelset_candidate",
                    }
                )
                global_checks = dict(selected_bundle["checks"])
                global_checks.update(
                    {
                        "raw_lddmm_shoot_not_inverted": shot_quality[
                            "inverted_tetrahedra"
                        ]
                        == 0,
                        "old_irregular_base_completely_removed": bool(
                            plane["complete_cut_within_bound"]
                        ),
                    }
                )
                selected_bundle["global_checks"] = global_checks
                status = "ACCEPTED" if all(global_checks.values()) else "REJECTED"
                write_unstructured_grid(volume_path, selected_bundle["grid"])
                write_polydata(surface_path, selected_bundle["surface"])
            else:
                error = (
                    "No topology-valid plane produced an acceptable MMG3D "
                    "level-set mesh. See cut_search.levelset_attempts for the "
                    "exact failures."
                )
        except Exception as exc:
            error = str(exc)

        outputs = {
            "generated_momenta": str(momentum_path),
            "generated_surface": (
                str(surface_path) if surface_path.is_file() else None
            ),
            "generated_volume": str(volume_path) if volume_path.is_file() else None,
        }
        volume_quality = (
            selected_bundle["volume_quality"]
            if selected_bundle is not None
            else None
        )
        surface_quality = (
            selected_bundle["surface_quality"]
            if selected_bundle is not None
            else None
        )
        row = {
            "sample": sample_id,
            "status": status,
            "plane_offset_shift": (
                selected_plane["selected_shift_from_ct55"]
                if selected_plane is not None
                else None
            ),
            "extra_inward_cut_depth": (
                selected_plane["quality_search_extra_inward_depth"]
                if selected_plane is not None
                else None
            ),
            "flatness_error": (
                surface_quality["maximum_basal_plane_error"]
                if surface_quality is not None
                else None
            ),
            "minimum_scaled_jacobian": (
                volume_quality["minimum_scaled_jacobian"]
                if volume_quality is not None
                else None
            ),
            "p01_scaled_jacobian": (
                volume_quality["p01_scaled_jacobian"]
                if volume_quality is not None
                else None
            ),
            "minimum_dihedral_angle": (
                volume_quality["minimum_dihedral_angle"]
                if volume_quality is not None
                else None
            ),
            "tetrahedra": (
                volume_quality["tetrahedra"] if volume_quality is not None else None
            ),
            "surface_triangles": (
                surface_quality["triangles"] if surface_quality is not None else None
            ),
            "surface": outputs["generated_surface"],
            "volume": outputs["generated_volume"],
            "momenta": str(momentum_path),
        }
        report: dict[str, Any] = {
            "status": status,
            "sample": sample_id,
            "generation_version": GENERATION_VERSION,
            "clipping_version": CLIPPING_VERSION,
            "template": "ct_case_0055",
            "method": "unchanged_momenta_MMG3D_affine_levelset_protected_collar",
            "scale_performed": False,
            "momentum_refinement_performed": False,
            "anatomical_surface_smoothing_performed": False,
            "direct_tetrahedron_clipping_performed": False,
            "surface_clip_performed": False,
            "surface_prescreen_performed": True,
            "implicit_levelset_cut_performed": True,
            "retetrahedralization_performed": True,
            "global_retetrahedralization_performed": False,
            "local_levelset_remeshing_performed": True,
            "local_quality_polish_performed": bool(
                selected_bundle is not None
                and selected_bundle.get("quality_polish_attempts")
            ),
            "selected_plane": selected_plane,
            "cut_search": cut_search,
            "shot_volume_relative_quality": shot_quality,
            "shot_volume_absolute_quality": shot_absolute_quality,
            "selected_remesh": (
                {
                    "levelset": selected_bundle["levelset"],
                    "levelset_insertion": selected_bundle[
                        "levelset_insertion"
                    ],
                    "levelset_mode_attempts": selected_bundle[
                        "levelset_mode_attempts"
                    ],
                    "quality_polish_attempts": selected_bundle[
                        "quality_polish_attempts"
                    ],
                    "quality_polish": selected_bundle.get("quality_polish"),
                    "quality_polish_collar": selected_bundle.get(
                        "quality_polish_collar"
                    ),
                    "cap_projection": selected_bundle["cap_projection"],
                    "protection": selected_bundle["protection"],
                    "frozen_region": selected_bundle["frozen_region"],
                    "volume_metadata": selected_bundle["volume_metadata"],
                    "anatomical_label_transfer": selected_bundle[
                        "anatomical_label_transfer"
                    ],
                    "reference_surface_quality": selected_bundle[
                        "reference_surface_quality"
                    ],
                    "reference_cap_topology": selected_bundle[
                        "reference_cap_topology"
                    ],
                    "volume_quality": selected_bundle["volume_quality"],
                    "surface_quality": selected_bundle["surface_quality"],
                    "final_cap_topology": selected_bundle["final_cap_topology"],
                    "surface_fidelity": selected_bundle["surface_fidelity"],
                    "surface_fidelity_tolerance": selected_bundle[
                        "surface_fidelity_tolerance"
                    ],
                    "surface_fidelity_p99_tolerance": selected_bundle[
                        "surface_fidelity_p99_tolerance"
                    ],
                    "surface_fidelity_rms_tolerance": selected_bundle[
                        "surface_fidelity_rms_tolerance"
                    ],
                    "checks": selected_bundle["global_checks"],
                }
                if selected_bundle is not None
                else None
            ),
            "error": error,
            "outputs": outputs,
            "summary_row": row,
        }
        write_json(report_path, report)
        rows.append(row)
        if selected_bundle is None:
            print(f"  REJECTED: {error}", flush=True)
        else:
            failed_checks = [
                name
                for name, passed in selected_bundle["global_checks"].items()
                if not passed
            ]
            print(
                f"  {status}: shift={row['plane_offset_shift']:+.3f} mm; "
                f"flat={row['flatness_error']:.3g} mm; "
                f"min/p01 SJ={row['minimum_scaled_jacobian']:.4g}/"
                f"{row['p01_scaled_jacobian']:.4g}; "
                f"min angle={row['minimum_dihedral_angle']:.3g} deg; "
                f"tets={row['tetrahedra']:,}",
                flush=True,
            )
            if failed_checks:
                print(
                    "  failed checks: " + ", ".join(failed_checks),
                    flush=True,
                )

    summary_csv = args.output_dir / "generation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "generation_complete",
        "processed": len(rows),
        "accepted": sum(row["status"] == "ACCEPTED" for row in rows),
        "rejected": sum(row["status"] == "REJECTED" for row in rows),
        "generation_version": GENERATION_VERSION,
        "clipping_version": CLIPPING_VERSION,
        "sampling": sampling_diagnostics,
        "template_surface": str(args.template_surface),
        "template_volume": str(args.template_volume),
        "template_volume_sha256": sha256(args.template_volume),
        "template_surface_to_volume_maximum_error": template_surface_volume_error,
        "mmg3d": {
            "executable": mmg_executable,
            "help_probe": mmg_probe,
            "mode": "affine_levelset_then_optional_local_optimization",
            "collar_width_mm": args.collar_width,
            "collar_rings": args.collar_rings,
            "wall_release_width_mm": args.wall_release_width,
            "wall_release_is_automatic": args.wall_release_width == 0,
            "quality_polish_passes": args.mmg_quality_passes,
            "hausdorff_tolerance_mm": args.mmg_hausdorff,
            "gradation": args.mmg_gradation,
        },
        "quality_requirements": {
            "flat_tolerance": args.flat_tolerance,
            "minimum_scaled_jacobian": args.minimum_scaled_jacobian,
            "minimum_p01_scaled_jacobian": args.minimum_p01_scaled_jacobian,
            "minimum_dihedral_angle": args.minimum_dihedral_angle,
            "expected_basal_contours": args.expected_basal_contours,
        },
        "rows": rows,
    }
    write_json(args.output_dir / "generation_summary.json", summary)
    print(
        f"\nGeneration complete: {summary['accepted']}/{summary['processed']} "
        f"accepted. Review {summary_csv} and {reports_dir}."
    )
    return 0 if summary["rejected"] == 0 else 2


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--_tetgen-worker":
        raise SystemExit(_tetgen_worker_main(sys.argv[2:]))
    try:
        raise SystemExit(levelset_generation_main())
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        ValueError,
        RuntimeError,
        FloatingPointError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

