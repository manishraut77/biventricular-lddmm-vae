#!/usr/bin/env python3
"""Fit one compact generative graph VAE on all 61 CT55 momentum fields.

This script is specific to the 61-subject CT55 pipeline:

* every input is an unchanged ``*_momenta.npz`` registration result;
* all subjects must use CT case 55, the same 1404 ordered control points,
  normalization frame, shooting step count, and registration version;
* registration reports must have passed unless explicitly skipped;
* geometry is never rescaled.  Coordinate/momentum standardization below is
  only an invertible numerical transform inside the neural network.

Every one of the 61 subjects is training data. Hyperparameters are fixed before
the fit. Normalization is computed from all 61 subjects because all 61 belong
to the declared training population, and is stored exactly for inverse scaling.

The loss combines normalized reconstruction error, KL warm-up with per-latent
free bits, and a DIP-VAE-II-style aggregate-posterior moment penalty. The last
term encourages the population posterior to match N(0,I), making prior samples
more useful while retaining a compact network suitable for a 61-case cohort.

This script deliberately does not call decoded fields "accepted hearts".
Strict basal flatness is an endpoint condition and must be enforced and checked
during the later decode -> constrained-refinement -> LDDMM-shooting stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Install it with: python3 -m pip install torch"
    ) from exc


FORMAT_VERSION = 1
MOMENTUM_NAME = re.compile(r"^(\d+)_momenta\.npz$")


@dataclass(frozen=True)
class Cohort:
    subject_ids: list[str]
    source_files: list[str]
    control_points: np.ndarray
    momenta: np.ndarray
    origin: np.ndarray
    shooting_scale: float
    shooting_steps: int
    template: str
    registration_version: str
    deformation_kernel_width: float
    plane_normals: np.ndarray
    plane_offsets: np.ndarray
    report_plane_errors: np.ndarray
    report_mean_distances: np.ndarray
    report_p95_distances: np.ndarray
    control_points_sha256: str
    momenta_sha256: str


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--momenta-dir",
        type=Path,
        default=Path("05.2_Momenta"),
        help="Folder containing 01_momenta.npz through 61_momenta.npz.",
    )
    parser.add_argument(
        "--registration-dir",
        type=Path,
        default=None,
        help=(
            "Optional folder containing ct_case_XXXX/report.json files for "
            "additional registration-QA checks. Training requires only the "
            "validated momentum NPZ files when this option is omitted."
        ),
    )
    parser.add_argument("--expected-count", type=int, default=61)
    parser.add_argument("--expected-control-points", type=int, default=1404)
    parser.add_argument("--template", default="ct_case_0055")
    parser.add_argument(
        "--kernel-width",
        type=float,
        default=16.0,
        help="LDDMM deformation-kernel width used by registration and shooting.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser(
        "audit", help="Validate all momentum files and registration reports only."
    )
    add_data_arguments(audit)
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("06.1_CT55GraphVAE/cohort_audit.json"),
    )

    train = commands.add_parser("train", help="fit the graph VAE on all subjects")
    add_data_arguments(train)
    train.add_argument(
        "--output-dir", type=Path, default=Path("06.1_CT55GraphVAE")
    )
    train.add_argument("--latent-dim", type=int, default=8)
    train.add_argument("--hidden-dim", type=int, default=48)
    train.add_argument("--neighbors", type=int, default=12)
    train.add_argument("--dropout", type=float, default=0.10)
    train.add_argument("--epochs", type=int, default=1800)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument(
        "--beta",
        type=float,
        default=5e-4,
        help=(
            "Final KL weight. Its numeric scale depends on this script's "
            "sum-over-latent-dimensions KL convention."
        ),
    )
    train.add_argument("--kl-warmup-epochs", type=int, default=600)
    train.add_argument("--free-nats", type=float, default=0.10)
    train.add_argument("--moment-weight", type=float, default=0.02)
    train.add_argument("--input-noise", type=float, default=0.005)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing checkpoint in --output-dir.",
    )
    train.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    return parser.parse_args()


def scalar_string(array: np.ndarray, name: str, path: Path) -> str:
    value = np.asarray(array)
    if value.size != 1:
        raise ValueError(f"{path}: {name} must be scalar, got shape {value.shape}")
    return str(value.reshape(-1)[0])


def scalar_float(array: np.ndarray, name: str, path: Path) -> float:
    value = np.asarray(array)
    if value.size != 1:
        raise ValueError(f"{path}: {name} must be scalar, got shape {value.shape}")
    result = float(value.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"{path}: {name} is not finite")
    return result


def update_hash(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))


def load_cohort(args: argparse.Namespace) -> Cohort:
    if args.expected_count < 3:
        raise ValueError("--expected-count must be at least 3")
    if args.expected_control_points < 2:
        raise ValueError("--expected-control-points must be at least 2")
    if args.kernel_width <= 0:
        raise ValueError("--kernel-width must be positive")
    if not args.momenta_dir.is_dir():
        raise FileNotFoundError(f"Momenta directory not found: {args.momenta_dir}")
    reports_checked = args.registration_dir is not None
    if reports_checked and not args.registration_dir.is_dir():
        raise FileNotFoundError(
            f"Registration directory not found: {args.registration_dir}"
        )

    indexed_paths: list[tuple[int, Path]] = []
    unexpected_npz: list[str] = []
    for path in args.momenta_dir.glob("*.npz"):
        match = MOMENTUM_NAME.fullmatch(path.name)
        if match:
            indexed_paths.append((int(match.group(1)), path))
        else:
            unexpected_npz.append(path.name)
    indexed_paths.sort(key=lambda item: item[0])
    expected_indices = list(range(1, args.expected_count + 1))
    actual_indices = [index for index, _ in indexed_paths]
    if actual_indices != expected_indices:
        missing = sorted(set(expected_indices).difference(actual_indices))
        duplicates = sorted(
            index for index in set(actual_indices) if actual_indices.count(index) > 1
        )
        raise ValueError(
            "Momentum filenames must form the exact sequence "
            f"01..{args.expected_count:02d}. Missing={missing}; duplicates={duplicates}; "
            f"found={actual_indices}"
        )
    if unexpected_npz:
        print(
            "NOTE: ignored NPZ files that do not match NN_momenta.npz: "
            + ", ".join(sorted(unexpected_npz)),
            file=sys.stderr,
        )

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
        "registration_version",
    }
    subject_ids: list[str] = []
    source_files: list[str] = []
    momenta_list: list[np.ndarray] = []
    plane_normals: list[np.ndarray] = []
    plane_offsets: list[float] = []
    report_plane_errors: list[float] = []
    report_mean_distances: list[float] = []
    report_p95_distances: list[float] = []
    reference_control: np.ndarray | None = None
    reference_origin: np.ndarray | None = None
    reference_scale: float | None = None
    reference_steps: int | None = None
    reference_version: str | None = None
    momenta_digest = hashlib.sha256()

    for sequence_number, path in indexed_paths:
        with np.load(path, allow_pickle=False) as data:
            missing_keys = required.difference(data.files)
            if missing_keys:
                raise ValueError(f"{path}: missing arrays {sorted(missing_keys)}")
            control = np.asarray(data["control_points"], dtype=np.float64)
            momentum = np.asarray(data["momenta"], dtype=np.float64)
            origin = np.asarray(data["origin"], dtype=np.float64)
            scale = scalar_float(data["scale"], "scale", path)
            steps_value = scalar_float(data["steps"], "steps", path)
            steps = int(steps_value)
            normal = np.asarray(data["plane_normal"], dtype=np.float64)
            offset = scalar_float(data["plane_offset"], "plane_offset", path)
            template = scalar_string(data["template"], "template", path)
            target = scalar_string(data["target"], "target", path)
            version = scalar_string(
                data["registration_version"], "registration_version", path
            )

        expected_target = f"ct_case_{sequence_number:04d}"
        if target != expected_target:
            raise ValueError(
                f"{path}: target={target!r}; filename position requires {expected_target!r}"
            )
        if template != args.template:
            raise ValueError(
                f"{path}: template={template!r}; expected {args.template!r}"
            )
        expected_shape = (args.expected_control_points, 3)
        if control.shape != expected_shape or momentum.shape != expected_shape:
            raise ValueError(
                f"{path}: control/momentum shapes are {control.shape}/{momentum.shape}; "
                f"expected {expected_shape}/{expected_shape}"
            )
        if origin.shape != (3,) or normal.shape != (3,):
            raise ValueError(
                f"{path}: origin and plane_normal must both have shape (3,)"
            )
        if not all(
            np.isfinite(value).all()
            for value in (control, momentum, origin, normal)
        ):
            raise ValueError(f"{path}: non-finite values found")
        normal_length = float(np.linalg.norm(normal))
        if not np.isclose(normal_length, 1.0, atol=1e-6, rtol=0.0):
            raise ValueError(f"{path}: plane normal length is {normal_length}, not 1")
        if scale <= 0 or steps < 1 or steps_value != steps:
            raise ValueError(f"{path}: invalid shooting scale/steps {scale}/{steps_value}")

        if reference_control is None:
            reference_control = control.copy()
            reference_origin = origin.copy()
            reference_scale = scale
            reference_steps = steps
            reference_version = version
        else:
            if not np.array_equal(control, reference_control):
                maximum = float(np.max(np.abs(control - reference_control)))
                raise ValueError(
                    f"{path}: control points/order differ from the first file "
                    f"(maximum absolute difference {maximum:.6g})"
                )
            if not np.array_equal(origin, reference_origin):
                raise ValueError(f"{path}: normalization origin differs")
            if scale != reference_scale:
                raise ValueError(f"{path}: shooting scale differs ({scale} vs {reference_scale})")
            if steps != reference_steps:
                raise ValueError(f"{path}: shooting steps differ ({steps} vs {reference_steps})")
            if version != reference_version:
                raise ValueError(
                    f"{path}: registration version differs ({version!r} vs {reference_version!r})"
                )

        if reports_checked:
            report_path = args.registration_dir / target / "report.json"
            if not report_path.is_file():
                raise FileNotFoundError(f"Missing registration report: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "registration_checks_passed":
                raise ValueError(
                    f"{report_path}: status is {report.get('status')!r}, not a pass"
                )
            if report.get("template") != args.template or report.get("target") != target:
                raise ValueError(f"{report_path}: template/target metadata mismatch")
            if int(report.get("common_control_points", -1)) != args.expected_control_points:
                raise ValueError(f"{report_path}: common-control-point count mismatch")
            settings = report.get("settings", {})
            reported_kernel = float(settings.get("kernel_width", float("nan")))
            if not np.isclose(reported_kernel, args.kernel_width, atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"{report_path}: kernel_width={reported_kernel}; "
                    f"expected {args.kernel_width}"
                )
            report_plane_errors.append(float(report["maximum_plane_error"]))
            anatomical = report["non_basal_anatomical_distance"]
            report_mean_distances.append(float(anatomical["mean"]))
            report_p95_distances.append(float(anatomical["p95"]))
        else:
            report_plane_errors.append(float("nan"))
            report_mean_distances.append(float("nan"))
            report_p95_distances.append(float("nan"))

        subject_ids.append(target)
        source_files.append(str(path))
        momenta_list.append(momentum)
        plane_normals.append(normal)
        plane_offsets.append(offset)
        update_hash(momenta_digest, momentum)

    assert reference_control is not None
    assert reference_origin is not None
    assert reference_scale is not None
    assert reference_steps is not None
    assert reference_version is not None
    control_digest = hashlib.sha256()
    update_hash(control_digest, reference_control)
    momenta = np.stack(momenta_list, axis=0)
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Duplicate target IDs found")
    return Cohort(
        subject_ids=subject_ids,
        source_files=source_files,
        control_points=reference_control,
        momenta=momenta,
        origin=reference_origin,
        shooting_scale=reference_scale,
        shooting_steps=reference_steps,
        template=args.template,
        registration_version=reference_version,
        deformation_kernel_width=float(args.kernel_width),
        plane_normals=np.stack(plane_normals, axis=0),
        plane_offsets=np.asarray(plane_offsets, dtype=np.float64),
        report_plane_errors=np.asarray(report_plane_errors, dtype=np.float64),
        report_mean_distances=np.asarray(report_mean_distances, dtype=np.float64),
        report_p95_distances=np.asarray(report_p95_distances, dtype=np.float64),
        control_points_sha256=control_digest.hexdigest(),
        momenta_sha256=momenta_digest.hexdigest(),
    )


def finite_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
    }


def audit_dictionary(cohort: Cohort, reports_checked: bool) -> dict[str, Any]:
    field_rms = np.sqrt(np.mean(cohort.momenta**2, axis=(1, 2)))
    return {
        "status": "PASS",
        "subjects": len(cohort.subject_ids),
        "subject_ids": cohort.subject_ids,
        "source_files": cohort.source_files,
        "template": cohort.template,
        "control_points": list(cohort.control_points.shape),
        "momenta": list(cohort.momenta.shape),
        "origin": cohort.origin.tolist(),
        "shooting_scale": cohort.shooting_scale,
        "shooting_steps": cohort.shooting_steps,
        "deformation_kernel_width": cohort.deformation_kernel_width,
        "registration_version": cohort.registration_version,
        "control_points_sha256": cohort.control_points_sha256,
        "momenta_sha256": cohort.momenta_sha256,
        "momentum_field_rms": finite_summary(field_rms),
        "registration_reports_checked": reports_checked,
        "registration_maximum_plane_error": finite_summary(
            cohort.report_plane_errors
        ),
        "registration_anatomical_mean_distance": finite_summary(
            cohort.report_mean_distances
        ),
        "registration_anatomical_p95_distance": finite_summary(
            cohort.report_p95_distances
        ),
        "invariants": {
            "identical_ordered_control_points": True,
            "identical_origin_scale_steps": True,
            "no_anatomical_scale_normalization": True,
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_knn(control_points: np.ndarray, neighbors: int) -> np.ndarray:
    """Return exactly ``neighbors`` non-self neighbours for every node."""
    n_points = len(control_points)
    if not 1 <= neighbors < n_points:
        raise ValueError(f"--neighbors must be between 1 and {n_points - 1}")
    stable = control_points.astype(np.float64) - control_points.mean(axis=0)
    squared_norms = np.sum(stable**2, axis=1)
    squared_distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * stable @ stable.T
    )
    np.maximum(squared_distances, 0.0, out=squared_distances)
    np.fill_diagonal(squared_distances, np.inf)
    selected = np.argpartition(squared_distances, kth=neighbors - 1, axis=1)[
        :, :neighbors
    ]
    selected_distances = np.take_along_axis(squared_distances, selected, axis=1)
    ordering = np.argsort(selected_distances, axis=1)
    return np.take_along_axis(selected, ordering, axis=1).astype(np.int64)


class GraphConvolution(nn.Module):
    """Small spatial graph layer using separate self and neighbour maps."""

    def __init__(
        self, input_dim: int, output_dim: int, neighbor_indices: torch.Tensor
    ) -> None:
        super().__init__()
        self.register_buffer("neighbor_indices", neighbor_indices)
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
        neighbor_indices: torch.Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        self.graph = GraphConvolution(input_dim, output_dim, neighbor_indices)
        self.normalization = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim, bias=False)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        update = self.graph(features)
        update = torch.nn.functional.gelu(self.normalization(update))
        return self.skip(features) + self.dropout(update)


class GraphMomentaBetaVAE(nn.Module):
    """Graph encoder plus coordinate-conditioned graph decoder."""

    def __init__(
        self,
        normalized_control_points: torch.Tensor,
        neighbor_indices: torch.Tensor,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout_probability = dropout
        self.register_buffer("coordinates", normalized_control_points)
        self.encoder_1 = ResidualGraphBlock(
            6, hidden_dim, neighbor_indices, dropout
        )
        self.encoder_2 = ResidualGraphBlock(
            hidden_dim, hidden_dim, neighbor_indices, dropout
        )
        self.encoder_3 = ResidualGraphBlock(
            hidden_dim, hidden_dim, neighbor_indices, dropout
        )
        self.encoder_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_log_variance = nn.Linear(hidden_dim, latent_dim)

        positional_dimension = 3 + 3 * 2 * 3
        self.decoder_input = nn.Sequential(
            nn.Linear(positional_dimension + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.decoder_1 = ResidualGraphBlock(
            hidden_dim, hidden_dim, neighbor_indices, dropout
        )
        self.decoder_2 = ResidualGraphBlock(
            hidden_dim, hidden_dim, neighbor_indices, dropout
        )
        self.decoder_output = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.decoder_output.bias)

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

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, log_variance: torch.Tensor, stochastic: bool
    ) -> torch.Tensor:
        if not stochastic:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_variance)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        positions = self.positional_encoding().unsqueeze(0).expand(batch, -1, -1)
        latent_field = latent[:, None, :].expand(-1, positions.shape[1], -1)
        hidden = self.decoder_input(torch.cat((positions, latent_field), dim=-1))
        hidden = self.decoder_1(hidden)
        hidden = self.decoder_2(hidden)
        return self.decoder_output(hidden)

    def forward(
        self, normalized_momenta: torch.Tensor, stochastic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_variance = self.encode(normalized_momenta)
        latent = self.reparameterize(mu, log_variance, stochastic)
        return self.decode(latent), mu, log_variance


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)










def denoising_input(batch: torch.Tensor, noise: float) -> torch.Tensor:
    """Corrupt only the encoder input; reconstruction targets stay clean."""
    if noise <= 0:
        return batch
    return batch + noise * torch.randn_like(batch)




@torch.no_grad()


def make_loader(
    normalized_momenta: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    tensor = torch.from_numpy(normalized_momenta[indices])
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(tensor),
        batch_size=min(batch_size, len(indices)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=device.type == "cuda",
    )


def normalization_from_indices(
    control_points: np.ndarray, momenta: np.ndarray, indices: np.ndarray
) -> tuple[
    tuple[np.ndarray, float, np.ndarray, float, np.ndarray], np.ndarray
]:
    coordinate_center = control_points.mean(axis=0)
    coordinate_scale = float(
        np.sqrt(np.mean((control_points - coordinate_center) ** 2))
    )
    if not math.isfinite(coordinate_scale) or coordinate_scale <= 1e-12:
        raise ValueError("Control points have zero or invalid spatial scale")
    normalized_control = (
        (control_points - coordinate_center) / coordinate_scale
    ).astype(np.float32)

    mean_field = momenta[indices].mean(axis=0)
    residuals = momenta - mean_field
    momentum_scale = float(np.sqrt(np.mean(residuals[indices] ** 2)))
    if not math.isfinite(momentum_scale) or momentum_scale <= 1e-12:
        raise ValueError("Momentum fields have zero or invalid variance")
    normalized_momenta = (residuals / momentum_scale).astype(np.float32)
    return (
        coordinate_center,
        coordinate_scale,
        mean_field,
        momentum_scale,
        normalized_control,
    ), normalized_momenta


def new_model(
    normalized_control: np.ndarray,
    neighbor_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> GraphMomentaBetaVAE:
    return GraphMomentaBetaVAE(
        torch.from_numpy(normalized_control).to(device),
        torch.from_numpy(neighbor_indices).to(device),
        args.latent_dim,
        args.hidden_dim,
        args.dropout,
    ).to(device)










def reconstruct_all(
    model: GraphMomentaBetaVAE,
    normalized_momenta: np.ndarray,
    mean_field: np.ndarray,
    momentum_scale: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    reconstructions: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(normalized_momenta), batch_size):
            batch = torch.from_numpy(
                normalized_momenta[start : start + batch_size]
            ).to(device)
            mu, logvar = model.encode(batch)
            reconstructed = model.decode(mu)
            reconstructions.append(reconstructed.cpu().numpy())
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    reconstructed_normalized = np.concatenate(reconstructions, axis=0)
    latent_mu = np.concatenate(mus, axis=0)
    latent_logvar = np.concatenate(logvars, axis=0)
    reconstructed_physical = (
        reconstructed_normalized.astype(np.float64) * momentum_scale + mean_field
    )
    return (
        reconstructed_physical,
        reconstructed_normalized,
        latent_mu,
        latent_logvar,
    )


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty training history")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)




def run_audit(args: argparse.Namespace) -> int:
    cohort = load_cohort(args)
    audit = audit_dictionary(cohort, args.registration_dir is not None)
    write_json(args.output, audit)
    print(
        f"PASS: {len(cohort.subject_ids)} subjects; "
        f"momenta={cohort.momenta.shape}; template={cohort.template}; "
        f"control hash={cohort.control_points_sha256[:12]}"
    )
    print(f"Wrote {args.output}")
    return 0




def aggregate_posterior_penalty(
    mu: torch.Tensor, log_variance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match aggregate posterior first and second moments to N(0, I)."""
    mean = mu.mean(dim=0)
    centered = mu - mean
    covariance_of_means = centered.T @ centered / max(mu.shape[0], 1)
    posterior_variance = torch.exp(log_variance).mean(dim=0)
    covariance = covariance_of_means + torch.diag(posterior_variance)
    identity = torch.eye(mu.shape[1], device=mu.device, dtype=mu.dtype)
    mean_penalty = mean.square().mean()
    covariance_penalty = (covariance - identity).square().mean()
    return mean_penalty + covariance_penalty, mean, covariance


def validate_all_fit_arguments(args: argparse.Namespace, subjects: int) -> None:
    if subjects != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} subjects, loaded {subjects}")
    if not 1 <= args.latent_dim < subjects:
        raise ValueError(f"--latent-dim must be between 1 and {subjects - 1}")
    if args.hidden_dim < 16 or not 1 <= args.neighbors < args.expected_control_points:
        raise ValueError("Invalid hidden dimension or graph-neighbor count")
    if not 0 <= args.dropout < 0.8:
        raise ValueError("--dropout must be in [0, 0.8)")
    if args.epochs < 1 or not 1 <= args.kl_warmup_epochs <= args.epochs:
        raise ValueError("Invalid epoch count or KL warm-up length")
    if args.batch_size < 2 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    if args.beta < 0 or args.free_nats < 0 or args.moment_weight < 0:
        raise ValueError("KL and aggregate-posterior weights must be non-negative")
    if args.input_noise < 0:
        raise ValueError("--input-noise cannot be negative")


def train_all_subjects(
    normalized_momenta: np.ndarray,
    normalized_control: np.ndarray,
    neighbor_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[GraphMomentaBetaVAE, list[dict[str, float | int]]]:
    seed_everything(args.seed)
    model = new_model(normalized_control, neighbor_indices, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    indices = np.arange(len(normalized_momenta), dtype=np.int64)
    loader = make_loader(
        normalized_momenta, indices, args.batch_size, True, args.seed, device
    )
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = args.beta * min(1.0, epoch / args.kl_warmup_epochs)
        totals = np.zeros(6, dtype=np.float64)
        examples = 0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, mu, log_variance = model(
                denoising_input(batch, args.input_noise), stochastic=True
            )
            reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
            kl_by_dimension = -0.5 * (
                1.0 + log_variance - mu.square() - log_variance.exp()
            ).mean(dim=0)
            effective_kl = torch.clamp(kl_by_dimension, min=args.free_nats).sum()
            raw_kl = kl_by_dimension.sum()
            moment_penalty, _, _ = aggregate_posterior_penalty(mu, log_variance)
            loss = (
                reconstruction_loss
                + beta * effective_kl
                + args.moment_weight * moment_penalty
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()
            count = len(batch)
            examples += count
            totals += count * np.asarray(
                [
                    float(loss.detach()),
                    float(reconstruction_loss.detach()),
                    float(raw_kl.detach()),
                    float(effective_kl.detach()),
                    float(moment_penalty.detach()),
                    float(gradient_norm.detach()),
                ]
            )
        scheduler.step()
        averages = totals / examples
        row = {
            "epoch": epoch,
            "beta": beta,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "loss": averages[0],
            "normalized_mse": averages[1],
            "raw_kl": averages[2],
            "free_bits_kl": averages[3],
            "aggregate_moment_penalty": averages[4],
            "gradient_norm": averages[5],
        }
        history.append(row)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:4d}/{args.epochs}: MSE={averages[1]:.6f}; "
                f"KL={averages[2]:.5f}; moment={averages[4]:.5f}; "
                f"beta={beta:.6g}",
                flush=True,
            )
    return model, history


@torch.no_grad()
def prior_generation_diagnostic(
    model: GraphMomentaBetaVAE,
    count: int,
    latent_dim: int,
    mean_field: np.ndarray,
    momentum_scale: float,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    latent = torch.randn(count, latent_dim, generator=generator).to(device)
    decoded = model.decode(latent).cpu().numpy().astype(np.float64)
    physical = decoded * momentum_scale + mean_field
    return latent.cpu().numpy(), physical


def run_train_all(args: argparse.Namespace) -> int:
    cohort = load_cohort(args)
    subjects = len(cohort.subject_ids)
    validate_all_fit_arguments(args, subjects)
    checkpoint_path = args.output_dir / "ct55_graph_momenta_vae_all61.pt"
    protected = [
        checkpoint_path,
        args.output_dir / "metrics.json",
        args.output_dir / "training_history.csv",
    ]
    conflicts = [str(path) for path in protected if path.exists()]
    if conflicts and not args.overwrite:
        raise ValueError(
            "Outputs already exist; choose another --output-dir or pass --overwrite: "
            + ", ".join(conflicts)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_dictionary(cohort, args.registration_dir is not None)

    all_indices = np.arange(subjects, dtype=np.int64)
    normalization, normalized_momenta = normalization_from_indices(
        cohort.control_points, cohort.momenta, all_indices
    )
    coordinate_center, coordinate_scale, mean_field, momentum_scale, normalized_control = normalization
    print(f"Subjects:       {subjects} (all used for fitting)")
    print(f"Control points: {len(cohort.control_points)}")
    print(f"Momenta:        {cohort.momenta.shape}")
    print("Fit scope:      all 61 subjects")
    print("Scaling:        invertible network normalization only; anatomy unchanged")
    neighbor_indices = build_knn(cohort.control_points, args.neighbors)
    device = choose_device(args.device)
    print(f"Device:         {device}")

    model, history = train_all_subjects(
        normalized_momenta, normalized_control, neighbor_indices, args, device
    )
    model.eval()
    reconstructed, reconstructed_normalized, latent_mu, latent_logvar = reconstruct_all(
        model,
        normalized_momenta,
        mean_field,
        momentum_scale,
        device,
        args.batch_size,
    )
    per_subject_rmse = np.sqrt(
        np.mean((reconstructed - cohort.momenta) ** 2, axis=(1, 2))
    )
    posterior_sd = np.exp(0.5 * latent_logvar)
    latent_variance = np.var(latent_mu, axis=0, ddof=1)
    active_dimensions = int(np.sum(latent_variance > 1e-2))
    aggregate_mean = latent_mu.mean(axis=0)
    centered_mu = latent_mu - aggregate_mean
    aggregate_covariance = (
        centered_mu.T @ centered_mu / subjects
        + np.diag(np.exp(latent_logvar).mean(axis=0))
    )
    sampled_latent, sampled_momenta = prior_generation_diagnostic(
        model, 256, args.latent_dim, mean_field, momentum_scale, device, args.seed + 1
    )
    training_variance = float(np.mean(np.var(cohort.momenta, axis=0, ddof=1)))
    sampled_variance = float(np.mean(np.var(sampled_momenta, axis=0, ddof=1)))
    diversity_ratio = sampled_variance / max(training_variance, 1e-30)

    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "purpose": "generation model fitted on all 61 CT subjects",
        "model_class": "GraphMomentaBetaVAE",
        "model_state": state,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "neighbors": args.neighbors,
        "dropout": args.dropout,
        "control_points": torch.from_numpy(cohort.control_points.astype(np.float32)),
        "neighbor_indices": torch.from_numpy(neighbor_indices),
        "coordinate_center": torch.from_numpy(coordinate_center.astype(np.float32)),
        "coordinate_scale": coordinate_scale,
        "mean_momentum_field": torch.from_numpy(mean_field.astype(np.float32)),
        "momentum_scale": momentum_scale,
        "subject_ids": cohort.subject_ids,
        "fit_subject_count": subjects,
        "training_epochs": args.epochs,
        "beta": args.beta,
        "kl_warmup_epochs": args.kl_warmup_epochs,
        "free_nats_per_dimension": args.free_nats,
        "aggregate_moment_weight": args.moment_weight,
        "sampling_prior": "standard_normal",
        "template": cohort.template,
        "origin": torch.from_numpy(cohort.origin.astype(np.float32)),
        "shooting_scale": cohort.shooting_scale,
        "shooting_steps": cohort.shooting_steps,
        "deformation_kernel_width": cohort.deformation_kernel_width,
        "registration_version": cohort.registration_version,
        "control_points_sha256": cohort.control_points_sha256,
        "momenta_sha256": cohort.momenta_sha256,
        "seed": args.seed,
        "strict_flatness_guaranteed_by_decoder": False,
    }
    torch.save(checkpoint, checkpoint_path)
    write_history(args.output_dir / "training_history.csv", history)

    warnings = [
        "Decoded momenta do not guarantee a flat endpoint; constrained refinement, shooting, and QA remain required."
    ]
    if active_dimensions < args.latent_dim:
        warnings.append(
            f"Only {active_dimensions}/{args.latent_dim} latent dimensions passed the activity threshold."
        )
    if diversity_ratio < 0.5:
        warnings.append(
            "Prior-decoded momentum variance is below half the training variance; inspect KL/moment balance before large-scale generation."
        )
    metrics = {
        "status": "training_complete",
        "fit_scope": "all_61_subjects_no_holdout_no_model_comparison",
        "configuration": {
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "neighbors": args.neighbors,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "beta": args.beta,
            "kl_warmup_epochs": args.kl_warmup_epochs,
            "free_nats_per_dimension": args.free_nats,
            "aggregate_moment_weight": args.moment_weight,
            "input_noise": args.input_noise,
            "seed": args.seed,
        },
        "fit_diagnostics": {
            "momentum_rmse": finite_summary(per_subject_rmse),
            "active_latent_dimensions": active_dimensions,
            "latent_mu_mean": aggregate_mean.tolist(),
            "latent_mu_standard_deviation": np.std(latent_mu, axis=0, ddof=1).tolist(),
            "posterior_standard_deviation_mean": posterior_sd.mean(axis=0).tolist(),
            "aggregate_posterior_covariance": aggregate_covariance.tolist(),
            "prior_sample_to_training_variance_ratio": diversity_ratio,
        },
        "data": audit,
        "warnings": warnings,
    }
    write_json(args.output_dir / "metrics.json", metrics)
    print(f"Active latent dimensions: {active_dimensions}/{args.latent_dim}")
    print(f"Prior/training variance ratio: {diversity_ratio:.4f}")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Wrote {checkpoint_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "audit":
        return run_audit(args)
    if args.command == "train":
        return run_train_all(args)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
