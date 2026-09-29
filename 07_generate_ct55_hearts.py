#!/usr/bin/env python3
"""Generate CT55-based synthetic hearts with constrained basal-plane fitting.

For each requested heart this script:

1. samples a latent vector and decodes a physical 1404 x 3 momentum field;
2. shoots the raw decoded momentum field from CT55;
3. fits a sample-specific plane parallel to CT55's basal plane;
4. minimally refines the momentum field with an augmented-Lagrangian endpoint
   constraint so every labeled basal vertex finishes on that plane;
5. re-shoots the complete CT55 surface and tetrahedral volume; and
6. writes refined momenta, a labeled VTK surface, a VTU volume, and QA reports.

No cut, remeshing, or anatomical scaling is performed. CT55's original surface
and tetrahedral connectivity are preserved through the smooth LDDMM flow.

Requires: numpy, torch, vtk
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
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


GENERATION_VERSION = "ct55_graph_vae_lddmm_flatbase_generation_v1"
REFINEMENT_VERSION = "ct55_parallel_sample_plane_endpoint_refinement_v3"
CLIPPING_VERSION = "quality_aware_sample_specific_tetra_clip_v2"


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
            "Reject z/temperature outside this radius. For eight dimensions, "
            "4.5 removes only extreme normal-prior tail draws."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run by preserving completed sample folders. "
            "A partial folder without report.json still stops the run."
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--optimization-steps", type=int, default=8)
    parser.add_argument(
        "--preservation-points",
        type=int,
        default=2000,
        help="Non-basal surface vertices used to preserve the raw generated anatomy.",
    )
    parser.add_argument("--outer-iterations", type=int, default=6)
    parser.add_argument("--inner-iterations", type=int, default=30)
    parser.add_argument("--polish-blocks", type=int, default=2)
    parser.add_argument("--polish-iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--initial-penalty", type=float, default=10.0)
    parser.add_argument("--penalty-growth", type=float, default=10.0)
    parser.add_argument("--maximum-penalty", type=float, default=1e6)
    parser.add_argument("--anatomy-weight", type=float, default=1.0)
    parser.add_argument(
        "--basal-tangent-weight",
        type=float,
        default=1.0,
        help="Preserve the raw generated basal outline within its selected plane.",
    )
    parser.add_argument("--momentum-proximity-weight", type=float, default=0.1)
    parser.add_argument(
        "--plane-policy",
        choices=("sample-offset", "ct55-fixed"),
        default="sample-offset",
        help=(
            "sample-offset keeps CT55's normal but fits a bounded offset from "
            "each raw generated base; ct55-fixed reproduces the older policy."
        ),
    )
    parser.add_argument(
        "--maximum-plane-offset-shift",
        type=float,
        default=5.0,
        help=(
            "Maximum absolute sample-plane displacement from CT55 in mm. "
            "Used only with --plane-policy sample-offset."
        ),
    )
    parser.add_argument(
        "--flat-tolerance",
        type=float,
        default=0.05,
        help="Maximum final basal-plane error in the input coordinate units (mm).",
    )
    parser.add_argument(
        "--optimization-flat-fraction",
        type=float,
        default=0.5,
        help="Optimization aims below this fraction of --flat-tolerance.",
    )
    parser.add_argument("--maximum-refinement-rms", type=float, default=2.0)
    parser.add_argument("--minimum-triangle-area-ratio", type=float, default=0.02)
    parser.add_argument("--minimum-tetra-volume-ratio", type=float, default=0.02)
    parser.add_argument("--shooting-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    integer_positive = (
        "count",
        "threads",
        "optimization_steps",
        "preservation_points",
        "outer_iterations",
        "inner_iterations",
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
    if args.polish_blocks < 0 or args.polish_iterations < 1:
        parser.error("--polish-blocks must be nonnegative and --polish-iterations positive")
    positive = (
        "learning_rate",
        "initial_penalty",
        "penalty_growth",
        "maximum_penalty",
        "anatomy_weight",
        "basal_tangent_weight",
        "momentum_proximity_weight",
        "maximum_plane_offset_shift",
        "flat_tolerance",
        "optimization_flat_fraction",
        "maximum_refinement_rms",
        "minimum_triangle_area_ratio",
        "minimum_tetra_volume_ratio",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.optimization_flat_fraction > 1:
        parser.error("--optimization-flat-fraction cannot exceed 1")
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
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Sample q(z)=N^-1 sum_i q(z|x_i), optionally expanded about its mean."""
    normalized = ((training_momenta - mean_field) / momentum_scale).astype(
        np.float32
    )
    with torch.inference_mode():
        mu_tensor, log_variance_tensor = model.encode(torch.from_numpy(normalized))
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
        decoded_normalized = model.decode(torch.from_numpy(latent)).cpu().numpy()
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


def main() -> int:
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
    quality_filter = vtk.vtkMeshQuality()
    quality_filter.SetInputData(grid)
    quality_filter.SetTetQualityMeasureToScaledJacobian()
    quality_filter.SaveCellQualityOn()
    quality_filter.Update()
    quality_array = quality_filter.GetOutput().GetCellData().GetArray("Quality")
    if quality_array is None:
        raise RuntimeError("VTK did not produce tetrahedral quality values")
    quality = np.asarray(vtk_to_numpy(quality_array), dtype=np.float64)
    finite = np.isfinite(quality)
    return {
        "points": int(grid.GetNumberOfPoints()),
        "tetrahedra": int(grid.GetNumberOfCells()),
        "nonfinite_scaled_jacobians": int(np.count_nonzero(~finite)),
        "nonpositive_scaled_jacobians": int(np.count_nonzero(quality[finite] <= 0)),
        "minimum_scaled_jacobian": float(np.min(quality[finite])) if np.any(finite) else None,
        "p01_scaled_jacobian": float(np.percentile(quality[finite], 1)) if np.any(finite) else None,
        "degenerate_tetrahedra": int(np.count_nonzero(absolute_volume <= volume_tolerance)),
        "minimum_absolute_volume": float(np.min(absolute_volume)),
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


def quality_aware_clipping_main() -> int:
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
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
