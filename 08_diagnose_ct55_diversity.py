#!/usr/bin/env python3
"""Diagnose where CT55 graph-VAE diversity is lost without changing any model.

The script compares the 61 training momentum fields with deterministic VAE
reconstructions, several latent sampling distributions, and (when available)
already generated raw/refined momentum fields.  It works entirely in momentum
space, so it is fast and does not alter meshes, checkpoints, or registrations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("06.1_CT55GraphVAE/ct55_graph_momenta_vae_all61.pt"),
    )
    parser.add_argument("--momenta-dir", type=Path, default=Path("05.2_Momenta"))
    parser.add_argument(
        "--refined-dir",
        type=Path,
        default=Path("07.1_Generated/RefinedMomentas"),
        help="Optional generated refined-momenta directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("08.1_DiversityDiagnostics")
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.samples < 61:
        parser.error("--samples must be at least 61")
    return args


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
    def __init__(self, input_dim: int, output_dim: int, neighbors: torch.Tensor,
                 dropout: float) -> None:
        super().__init__()
        self.graph = GraphConvolution(input_dim, output_dim, neighbors)
        self.normalization = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = (nn.Identity() if input_dim == output_dim
                     else nn.Linear(input_dim, output_dim, bias=False))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        update = torch.nn.functional.gelu(self.normalization(self.graph(features)))
        return self.skip(features) + self.dropout(update)


class GraphMomentaBetaVAE(nn.Module):
    """Architecture used by 06_fit_ct55_graph_vae_all.py."""

    def __init__(self, coordinates: torch.Tensor, neighbors: torch.Tensor,
                 latent_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.register_buffer("coordinates", coordinates)
        self.encoder_1 = ResidualGraphBlock(6, hidden_dim, neighbors, dropout)
        self.encoder_2 = ResidualGraphBlock(hidden_dim, hidden_dim, neighbors, dropout)
        self.encoder_3 = ResidualGraphBlock(hidden_dim, hidden_dim, neighbors, dropout)
        self.encoder_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_log_variance = nn.Linear(hidden_dim, latent_dim)
        positional_dim = 3 + 3 * 2 * 3
        self.decoder_input = nn.Sequential(
            nn.Linear(positional_dim + latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.decoder_1 = ResidualGraphBlock(hidden_dim, hidden_dim, neighbors, dropout)
        self.decoder_2 = ResidualGraphBlock(hidden_dim, hidden_dim, neighbors, dropout)
        self.decoder_output = nn.Linear(hidden_dim, 3)

    def positional_encoding(self) -> torch.Tensor:
        values = [self.coordinates]
        for frequency in (1.0, 2.0, 4.0):
            angle = math.pi * frequency * self.coordinates
            values.extend((torch.sin(angle), torch.cos(angle)))
        return torch.cat(values, dim=-1)

    def encode(self, momenta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = self.coordinates.unsqueeze(0).expand(momenta.shape[0], -1, -1)
        hidden = torch.cat((coordinates, momenta), dim=-1)
        hidden = self.encoder_1(hidden)
        hidden = self.encoder_2(hidden)
        hidden = self.encoder_3(hidden)
        pooled = torch.cat((hidden.mean(1), hidden.amax(1)), dim=-1)
        encoded = self.encoder_head(pooled)
        return self.to_mu(encoded), torch.clamp(self.to_log_variance(encoded), -10, 6)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        positions = self.positional_encoding().unsqueeze(0).expand(latent.shape[0], -1, -1)
        field = latent[:, None, :].expand(-1, positions.shape[1], -1)
        hidden = self.decoder_input(torch.cat((positions, field), dim=-1))
        return self.decoder_output(self.decoder_2(self.decoder_1(hidden)))


def load_checkpoint(path: Path) -> tuple[dict[str, Any], GraphMomentaBetaVAE]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required = ("model_state", "control_points", "neighbor_indices",
                "coordinate_center", "coordinate_scale", "mean_momentum_field",
                "momentum_scale", "latent_dim", "hidden_dim", "dropout")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing: {missing}")
    control = np.asarray(checkpoint["control_points"], dtype=np.float64)
    center = np.asarray(checkpoint["coordinate_center"], dtype=np.float64)
    scale = float(checkpoint["coordinate_scale"])
    normalized_control = torch.from_numpy(((control - center) / scale).astype(np.float32))
    model = GraphMomentaBetaVAE(
        normalized_control,
        torch.as_tensor(checkpoint["neighbor_indices"], dtype=torch.long),
        int(checkpoint["latent_dim"]),
        int(checkpoint["hidden_dim"]),
        float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return checkpoint, model


def load_training_momenta(directory: Path, expected_control: np.ndarray) -> np.ndarray:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    records: list[tuple[str, np.ndarray]] = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "momenta" not in data:
                continue
            momentum = np.asarray(data["momenta"], dtype=np.float64)
            if momentum.shape != expected_control.shape:
                raise ValueError(f"{path}: momentum shape {momentum.shape}")
            if "control_points" in data:
                saved_control = np.asarray(data["control_points"], dtype=np.float64)
                if saved_control.shape != expected_control.shape:
                    raise ValueError(
                        f"{path}: control-point shape {saved_control.shape}; "
                        f"expected {expected_control.shape}"
                    )
                # The training checkpoint intentionally stores geometry as
                # float32, whereas registration NPZ files commonly retain
                # float64.  At cardiac-coordinate magnitudes that conversion
                # can introduce several micrometres of roundoff.  This absolute
                # tolerance accepts only float32 quantization; it remains far
                # below the 8 mm control-point spacing and therefore still
                # catches reordered or genuinely different control grids.
                maximum_control_error = float(
                    np.max(np.abs(saved_control - expected_control))
                )
                if maximum_control_error > 5e-5:
                    raise ValueError(
                        f"{path}: control-point grid differs from checkpoint "
                        f"(maximum coordinate error={maximum_control_error:.6g})"
                    )
            records.append((path.name, momentum))
    if len(records) != 61:
        raise ValueError(f"Expected 61 momentum files, found {len(records)}")
    result = np.stack([value for _, value in records])
    if not np.isfinite(result).all():
        raise ValueError("Training momenta contain non-finite values")
    return result


def decode(model: GraphMomentaBetaVAE, latent: np.ndarray, mean: np.ndarray,
           scale: float, batch: int = 64) -> np.ndarray:
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(latent), batch):
            value = model.decode(torch.from_numpy(latent[start:start + batch].astype(np.float32)))
            pieces.append(value.cpu().numpy())
    return np.concatenate(pieces).astype(np.float64) * scale + mean


def field_variance(fields: np.ndarray) -> float:
    return float(np.mean(np.var(fields, axis=0, ddof=1)))


def pairwise_rmse(fields: np.ndarray, maximum: int = 512) -> dict[str, float]:
    fields = fields[:maximum].reshape(min(len(fields), maximum), -1)
    square = np.sum(fields * fields, axis=1)
    distance2 = np.maximum(square[:, None] + square[None, :] - 2 * fields @ fields.T, 0)
    values = np.sqrt(distance2[np.triu_indices(len(fields), 1)] / fields.shape[1])
    return {
        "minimum": float(np.min(values)), "median": float(np.median(values)),
        "mean": float(np.mean(values)), "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def summarize(name: str, fields: np.ndarray, training_variance: float) -> dict[str, Any]:
    variance = field_variance(fields)
    return {
        "name": name, "count": len(fields), "field_variance": variance,
        "variance_ratio_to_training": variance / max(training_variance, 1e-30),
        "pairwise_momentum_rmse": pairwise_rmse(fields),
    }


def generated_summary(directory: Path, training_variance: float) -> dict[str, Any] | None:
    if not directory.is_dir():
        return None
    raw, refined = [], []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "raw_decoded_momenta" in data and "momenta" in data:
                raw.append(np.asarray(data["raw_decoded_momenta"], dtype=np.float64))
                refined.append(np.asarray(data["momenta"], dtype=np.float64))
    if len(raw) < 2:
        return None
    raw_array, refined_array = np.stack(raw), np.stack(refined)
    change = np.sqrt(np.mean((refined_array - raw_array) ** 2, axis=(1, 2)))
    return {
        "raw_generated": summarize("existing_raw_generated", raw_array, training_variance),
        "refined_generated": summarize("existing_refined_generated", refined_array, training_variance),
        "refined_to_raw_variance_ratio": field_variance(refined_array) / max(field_variance(raw_array), 1e-30),
        "per_sample_refinement_momentum_rmse": {
            "median": float(np.median(change)), "maximum": float(np.max(change))
        },
    }


def main() -> int:
    args = arguments()
    rng = np.random.default_rng(args.seed)
    checkpoint, model = load_checkpoint(args.checkpoint)
    control = np.asarray(checkpoint["control_points"], dtype=np.float64)
    training = load_training_momenta(args.momenta_dir, control)
    mean = np.asarray(checkpoint["mean_momentum_field"], dtype=np.float64)
    momentum_scale = float(checkpoint["momentum_scale"])
    normalized = ((training - mean) / momentum_scale).astype(np.float32)
    with torch.inference_mode():
        mu_tensor, logvar_tensor = model.encode(torch.from_numpy(normalized))
        reconstruction_normalized = model.decode(mu_tensor).cpu().numpy()
    mu = mu_tensor.cpu().numpy().astype(np.float64)
    logvar = logvar_tensor.cpu().numpy().astype(np.float64)
    reconstruction = reconstruction_normalized.astype(np.float64) * momentum_scale + mean
    training_variance = field_variance(training)

    latent_dim = mu.shape[1]
    candidates: dict[str, np.ndarray] = {}
    for temperature in (1.0, 1.25, 2.0):
        latent = rng.standard_normal((args.samples, latent_dim)) * temperature
        candidates[f"standard_prior_T{temperature:g}"] = decode(
            model, latent, mean, momentum_scale
        )

    # The aggregate posterior follows the actually occupied latent regions.
    subject = rng.integers(0, len(mu), size=args.samples)
    epsilon = rng.standard_normal((args.samples, latent_dim))
    mixture_latent = mu[subject] + epsilon * np.exp(0.5 * logvar[subject])
    candidates["aggregate_posterior_mixture"] = decode(
        model, mixture_latent, mean, momentum_scale
    )
    # Expanded posterior means test whether diversity is available along learned axes.
    aggregate_mean = mu.mean(axis=0)
    expanded_latent = aggregate_mean + 1.5 * (mu[subject] - aggregate_mean)
    candidates["expanded_posterior_means_1p5"] = decode(
        model, expanded_latent, mean, momentum_scale
    )

    summaries = [summarize("training", training, training_variance),
                 summarize("deterministic_reconstruction", reconstruction, training_variance)]
    summaries.extend(summarize(name, fields, training_variance)
                     for name, fields in candidates.items())

    sensitivity = []
    for dimension in range(latent_dim):
        plus = aggregate_mean.copy(); plus[dimension] += 1
        minus = aggregate_mean.copy(); minus[dimension] -= 1
        decoded = decode(model, np.stack((plus, minus)), mean, momentum_scale)
        sensitivity.append(float(np.sqrt(np.mean((decoded[0] - decoded[1]) ** 2)) / 2))

    generated = generated_summary(args.refined_dir, training_variance)
    reconstruction_ratio = summaries[1]["variance_ratio_to_training"]
    prior_ratio = next(x for x in summaries if x["name"] == "standard_prior_T1")["variance_ratio_to_training"]
    recommendations = []
    if reconstruction_ratio < 0.75:
        recommendations.append(
            "The decoder loses substantial cohort variance even on training reconstructions; retraining/capacity is the primary bottleneck."
        )
    if prior_ratio < 0.75 * reconstruction_ratio:
        recommendations.append(
            "Standard-normal sampling loses more variance than reconstruction; the learned aggregate posterior does not match the nominal prior well."
        )
    if generated and generated["refined_to_raw_variance_ratio"] < 0.8:
        recommendations.append(
            "Flat-base refinement removes over 20% of raw generated momentum variance; strengthen non-basal preservation or localize the constraint."
        )
    if not recommendations:
        recommendations.append(
            "Momentum-space diversity is broadly preserved; inspect shot surfaces with anatomical measurements because visual comparison may hide internal variation."
        )

    report = {
        "status": "diagnostic_complete", "checkpoint": str(args.checkpoint),
        "training_subjects": len(training), "control_points": len(control),
        "latent_dim": latent_dim,
        "latent": {
            "posterior_mean": mu.mean(axis=0).tolist(),
            "posterior_mean_standard_deviation": mu.std(axis=0, ddof=1).tolist(),
            "posterior_sd_mean": np.exp(0.5 * logvar).mean(axis=0).tolist(),
            "decoder_one_sigma_momentum_sensitivity": sensitivity,
        },
        "cohorts": summaries, "existing_generation": generated,
        "interpretation": recommendations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "diversity_diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "diversity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "name", "count", "field_variance", "variance_ratio_to_training",
            "pairwise_rmse_median", "pairwise_rmse_p95",
        ))
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                "name": row["name"], "count": row["count"],
                "field_variance": row["field_variance"],
                "variance_ratio_to_training": row["variance_ratio_to_training"],
                "pairwise_rmse_median": row["pairwise_momentum_rmse"]["median"],
                "pairwise_rmse_p95": row["pairwise_momentum_rmse"]["p95"],
            })

    print("\nDiversity ratios (training = 1.0)")
    for row in summaries:
        print(f"  {row['name']:<34} {row['variance_ratio_to_training']:.4f}")
    if generated:
        print(f"  {'existing_refined/raw':<34} {generated['refined_to_raw_variance_ratio']:.4f}")
    print("\nInterpretation")
    for item in recommendations:
        print(f"  - {item}")
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
