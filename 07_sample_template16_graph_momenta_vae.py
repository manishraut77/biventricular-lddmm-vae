#!/usr/bin/env python3
"""Sample momentum fields from a trained combined graph beta-VAE checkpoint.

This script performs inference only. It never trains or modifies model weights.
For a temperature sweep, the same base latent vectors are scaled at each
temperature, enabling paired comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install it with: python3 -m pip install torch") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("MomentumVAE/Template16GraphBetaVAE/graph_momenta_beta_vae.pt"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("LDDMMRegisteredTemplate16/trainingdata/momenta_dataset.npz"),
        help="Used only for nearest-training diagnostics.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("MomentumVAE/Template16GraphBetaVAE/sampled_momenta"),
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7])
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing sample files in the selected temperature folders.",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if any(not np.isfinite(t) or t <= 0 for t in args.temperatures):
        parser.error("Every temperature must be finite and positive")
    if len(set(args.temperatures)) != len(args.temperatures):
        parser.error("--temperatures contains duplicates")
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, neighbor_indices: torch.Tensor):
        super().__init__()
        self.register_buffer("neighbor_indices", neighbor_indices)
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        neighboring = features[:, self.neighbor_indices, :].mean(dim=2)
        return self.self_linear(features) + self.neighbor_linear(neighboring)


class GraphMomentaBetaVAE(nn.Module):
    def __init__(
        self,
        normalized_control_points: torch.Tensor,
        neighbor_indices: torch.Tensor,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.register_buffer("coordinates", normalized_control_points)

        self.encoder_1 = GraphConvolution(6, hidden_dim, neighbor_indices)
        self.encoder_2 = GraphConvolution(hidden_dim, hidden_dim, neighbor_indices)
        self.encoder_3 = GraphConvolution(hidden_dim, hidden_dim, neighbor_indices)
        self.encoder_norm_1 = nn.LayerNorm(hidden_dim)
        self.encoder_norm_2 = nn.LayerNorm(hidden_dim)
        self.encoder_norm_3 = nn.LayerNorm(hidden_dim)
        self.encoder_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
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
        self.decoder_graph_1 = GraphConvolution(hidden_dim, hidden_dim, neighbor_indices)
        self.decoder_graph_2 = GraphConvolution(hidden_dim, hidden_dim, neighbor_indices)
        self.decoder_norm_1 = nn.LayerNorm(hidden_dim)
        self.decoder_norm_2 = nn.LayerNorm(hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, 3)

    def positional_encoding(self) -> torch.Tensor:
        components = [self.coordinates]
        for frequency in (1.0, 2.0, 4.0):
            argument = math.pi * frequency * self.coordinates
            components.extend((torch.sin(argument), torch.cos(argument)))
        return torch.cat(components, dim=-1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        position = self.positional_encoding().unsqueeze(0).expand(batch_size, -1, -1)
        latent_field = latent.unsqueeze(1).expand(-1, position.shape[1], -1)
        hidden = self.decoder_input(torch.cat((position, latent_field), dim=-1))
        hidden = torch.nn.functional.gelu(
            self.decoder_norm_1(self.decoder_graph_1(hidden))
        )
        hidden = torch.nn.functional.gelu(
            self.decoder_norm_2(self.decoder_graph_2(hidden))
        )
        return self.decoder_output(hidden)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    required = {
        "model_state",
        "latent_dim",
        "hidden_dim",
        "control_points",
        "neighbor_indices",
        "coordinate_center",
        "coordinate_scale",
        "mean_momentum_field",
        "momentum_scale",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    return checkpoint


def temperature_slug(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"temperature_{text}"


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)

    control_points = checkpoint["control_points"].detach().cpu().numpy().astype(np.float64)
    center = checkpoint["coordinate_center"].detach().cpu().numpy()
    coordinate_scale = checkpoint["coordinate_scale"].detach().cpu().numpy()
    normalized_coordinates = ((control_points - center) / coordinate_scale).astype(
        np.float32
    )
    neighbor_indices = checkpoint["neighbor_indices"].detach().cpu().long()
    mean_field = (
        checkpoint["mean_momentum_field"].detach().cpu().numpy().astype(np.float64)
    )
    momentum_scale = (
        checkpoint["momentum_scale"].detach().cpu().numpy().astype(np.float64)
    )
    latent_dim = int(checkpoint["latent_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])

    model = GraphMomentaBetaVAE(
        torch.from_numpy(normalized_coordinates),
        neighbor_indices,
        latent_dim,
        hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    training_momenta = None
    subject_ids: list[str] = []
    if args.dataset.is_file():
        with np.load(args.dataset, allow_pickle=False) as dataset:
            training_momenta = np.asarray(dataset["momenta"], dtype=np.float64)
            subject_ids = [str(v) for v in dataset["subject_ids"].tolist()]

    latent_generator = torch.Generator(device="cpu")
    latent_generator.manual_seed(args.seed)
    base_latent_cpu = torch.randn(
        args.count, latent_dim, generator=latent_generator, dtype=torch.float32
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Device:           {device}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Samples/cohort:   {args.count}")
    print(f"Temperatures:     {args.temperatures}")
    print(f"Seed:             {args.seed}")

    summary: list[dict[str, object]] = []
    for temperature in args.temperatures:
        temperature = float(temperature)
        cohort_dir = args.output_root / temperature_slug(temperature)
        cohort_dir.mkdir(parents=True, exist_ok=True)
        expected_files = [
            cohort_dir / f"generated_{i:03d}_momenta.txt"
            for i in range(1, args.count + 1)
        ]
        collisions = [p for p in expected_files if p.exists()]
        if collisions and not args.overwrite:
            raise FileExistsError(
                f"{len(collisions)} outputs already exist in {cohort_dir}; "
                "use --overwrite only if replacement is intended"
            )

        latent_cpu = base_latent_cpu * temperature
        with torch.no_grad():
            decoded = model.decode(latent_cpu.to(device)).cpu().numpy()
        generated = decoded * momentum_scale + mean_field
        if generated.shape != (args.count, len(control_points), 3):
            raise RuntimeError(f"Unexpected generated shape: {generated.shape}")
        if not np.isfinite(generated).all():
            raise RuntimeError(f"Generated NaN/Inf at temperature {temperature:g}")

        rows: list[dict[str, object]] = []
        flattened = generated.reshape(args.count, -1)
        for index, (path, field) in enumerate(zip(expected_files, generated), start=1):
            np.savetxt(path, field, fmt="%.10e")
            row: dict[str, object] = {
                "sample": f"generated_{index:03d}",
                "temperature": temperature,
                "seed": args.seed,
            }
            for component in range(latent_dim):
                row[f"base_z_{component + 1}"] = float(
                    base_latent_cpu[index - 1, component]
                )
                row[f"scaled_z_{component + 1}"] = float(
                    latent_cpu[index - 1, component]
                )
            if training_momenta is not None:
                train_flat = training_momenta.reshape(len(training_momenta), -1)
                distances = np.sqrt(
                    np.mean((train_flat - flattened[index - 1][None, :]) ** 2, axis=1)
                )
                nearest = int(np.argmin(distances))
                row["nearest_training_subject"] = subject_ids[nearest]
                row["nearest_training_momentum_rmse"] = float(distances[nearest])
            rows.append(row)

        if args.count > 1:
            pairwise = np.sqrt(
                np.mean(
                    (flattened[:, None, :] - flattened[None, :, :]) ** 2,
                    axis=2,
                )
            )
            upper = pairwise[np.triu_indices(args.count, k=1)]
            diversity_min = float(upper.min())
            diversity_median = float(np.median(upper))
            diversity_max = float(upper.max())
        else:
            diversity_min = diversity_median = diversity_max = 0.0

        with (cohort_dir / "latent_samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        np.savez_compressed(
            cohort_dir / "sampled_momenta.npz",
            control_points=control_points.astype(np.float32),
            generated_momenta=generated.astype(np.float32),
            base_latent=base_latent_cpu.numpy(),
            scaled_latent=latent_cpu.numpy(),
            temperature=np.float32(temperature),
            seed=np.int64(args.seed),
        )
        metadata = {
            "checkpoint": str(args.checkpoint),
            "dataset": str(args.dataset),
            "temperature": temperature,
            "seed": args.seed,
            "count": args.count,
            "latent_dimension": latent_dim,
            "control_points": len(control_points),
            "paired_temperature_design": True,
            "pairwise_momentum_rmse_min": diversity_min,
            "pairwise_momentum_rmse_median": diversity_median,
            "pairwise_momentum_rmse_max": diversity_max,
        }
        (cohort_dir / "sampling_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        summary.append(metadata)
        print(
            f"temperature={temperature:g}: {cohort_dir}; "
            f"pairwise RMSE median={diversity_median:.6g}"
        )

    (args.output_root / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSampling complete. Model weights were not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        ValueError,
        RuntimeError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
