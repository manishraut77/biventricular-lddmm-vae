#!/usr/bin/env python3
"""Train a compact graph beta-VAE on Deformetrica control-point momenta.

Designed for the fixed-subject-16 combined LV+RV dataset in:
    LDDMMRegisteredTemplate16/trainingdata/momenta_dataset.npz

The common control points define a k-nearest-neighbour graph. Each subject is
one [control_points, 3] momentum field. The VAE uses graph convolutions to
encode local spatial relationships and a  graph decoder
to reconstruct or generate momentum fields.

All 20 subjects are used for fitting. Reported reconstruction errors are
training errors only;
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install it with: python3 -m pip install torch") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph beta-VAE for shared-grid LDDMM momentum fields."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("LDDMMRegisteredTemplate16/trainingdata/momenta_dataset.npz"),
        help="Combined-atlas NPZ containing subject_ids, control_points, and momenta.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("MomentumVAE/Template16GraphBetaVAE")
    )
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=250)
    parser.add_argument("--input-noise", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument(
        "--generated-samples",
        type=int,
        default=0,
        help=(
            "Legacy convenience option. Keep at 0 and use "
            "07_sample_template16_graph_momenta_vae.py after training."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_single_file(directory: Path, token: str) -> Path:
    matches = sorted(path for path in directory.glob("*.txt") if token in path.name)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*{token}*.txt' in {directory}; found {len(matches)}"
        )
    return matches[0]


def read_subject_ids(data_set_path: Path) -> list[str]:
    root = ET.parse(data_set_path).getroot()
    result: list[str] = []
    for element in root.iter():
        if element.tag.split("}")[-1] == "subject" and "id" in element.attrib:
            result.append(element.attrib["id"])
    if not result:
        raise ValueError(f"No subject IDs found in {data_set_path}")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate subject IDs found in {data_set_path}")
    return result


def read_deformation_kernel_width(model_path: Path) -> float | None:
    root = ET.parse(model_path).getroot()
    for element in root.iter():
        if element.tag.split("}")[-1] == "kernel-width" and element.text:
            try:
                return float(element.text.strip())
            except ValueError:
                continue
    return None


def read_momenta(path: Path, subject_count: int, control_point_count: int) -> np.ndarray:
    """Read both common Deformetrica text layouts.

    Supported layouts:
      1. Header: `subjects control_points dimension`, followed by numeric rows.
      2. Plain [subjects * control_points, 3] rows, with optional blank lines.
    """
    nonempty = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not nonempty:
        raise ValueError(f"Empty momenta file: {path}")

    header_tokens = nonempty[0].split()
    integer_header = (
        len(header_tokens) == 3
        and all(re.fullmatch(r"[+-]?\d+", token) for token in header_tokens)
    )
    if integer_header:
        n_subjects, n_points, dimension = map(int, header_tokens)
        values = np.loadtxt(nonempty[1:], dtype=np.float64)
        values = np.atleast_2d(values)
        if (n_subjects, n_points, dimension) != (
            subject_count, control_point_count, 3
        ):
            raise ValueError(
                "Momenta header is "
                f"{(n_subjects, n_points, dimension)}, expected "
                f"{(subject_count, control_point_count, 3)}"
            )
        if values.shape != (n_subjects * n_points, dimension):
            raise ValueError(
                f"Momenta header and numeric data disagree: {values.shape}"
            )
        return values.reshape(n_subjects, n_points, dimension)

    values = np.loadtxt(path, dtype=np.float64, comments="#")
    values = np.atleast_2d(values)
    expected = subject_count * control_point_count
    if values.shape != (expected, 3):
        raise ValueError(
            f"Momenta have shape {values.shape}; expected {(expected, 3)} "
            f"for {subject_count} subjects and {control_point_count} control points."
        )
    return values.reshape(subject_count, control_point_count, 3)


def build_knn(control_points: np.ndarray, neighbors: int, chunk_size: int = 256) -> np.ndarray:
    """Return [N, neighbors + 1] indices, including the node itself."""
    n_points = control_points.shape[0]
    if not 1 <= neighbors < n_points:
        raise ValueError(f"--neighbors must be between 1 and {n_points - 1}")
    # Translation does not change distances and centering avoids overflow or
    # cancellation when coordinates have a large global offset.
    stable_points = control_points.astype(np.float64) - control_points.mean(axis=0)
    squared_norms = np.sum(stable_points * stable_points, axis=1)
    result = np.empty((n_points, neighbors + 1), dtype=np.int64)
    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        block = stable_points[start:stop]
        distances = (
            np.sum(block * block, axis=1, keepdims=True)
            + squared_norms[None, :]
            - 2.0 * block @ stable_points.T
        )
        np.maximum(distances, 0.0, out=distances)
        selected = np.argpartition(distances, kth=neighbors, axis=1)[:, : neighbors + 1]
        selected_distances = np.take_along_axis(distances, selected, axis=1)
        ordering = np.argsort(selected_distances, axis=1)
        result[start:stop] = np.take_along_axis(selected, ordering, axis=1)
    return result


class GraphConvolution(nn.Module):
    """Parameter-efficient mean-neighbour graph convolution."""

    def __init__(self, input_dim: int, output_dim: int, neighbor_indices: torch.Tensor):
        super().__init__()
        self.register_buffer("neighbor_indices", neighbor_indices)
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        neighboring = features[:, self.neighbor_indices, :].mean(dim=2)
        return self.self_linear(features) + self.neighbor_linear(neighboring)


class GraphMomentaBetaVAE(nn.Module):
    """Graph encoder and coordinate-conditioned graph decoder."""

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

    def encode(self, normalized_momenta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = normalized_momenta.shape[0]
        coordinates = self.coordinates.unsqueeze(0).expand(batch_size, -1, -1)
        hidden = torch.cat((coordinates, normalized_momenta), dim=-1)
        hidden = torch.nn.functional.gelu(
            self.encoder_norm_1(self.encoder_1(hidden))
        )
        hidden = torch.nn.functional.gelu(
            self.encoder_norm_2(self.encoder_2(hidden))
        )
        hidden = torch.nn.functional.gelu(
            self.encoder_norm_3(self.encoder_3(hidden))
        )
        pooled = torch.cat((hidden.mean(dim=1), hidden.amax(dim=1)), dim=-1)
        encoded = self.encoder_head(pooled)
        return self.to_mu(encoded), self.to_log_variance(encoded)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, log_variance: torch.Tensor, stochastic: bool
    ) -> torch.Tensor:
        if not stochastic:
            return mu
        standard_deviation = torch.exp(0.5 * log_variance)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

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

    def forward(
        self, normalized_momenta: torch.Tensor, stochastic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_variance = self.encode(normalized_momenta)
        latent = self.reparameterize(mu, log_variance, stochastic)
        return self.decode(latent), mu, log_variance


def loss_components(
    reconstruction: torch.Tensor,
    truth: torch.Tensor,
    mu: torch.Tensor,
    log_variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstruction_loss = torch.mean((reconstruction - truth) ** 2)
    kl_loss = -0.5 * torch.mean(
        1.0 + log_variance - mu.square() - log_variance.exp()
    )
    return reconstruction_loss, kl_loss


@torch.no_grad()
def evaluate(
    model: GraphMomentaBetaVAE,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    reconstruction_sum = 0.0
    kl_sum = 0.0
    sample_count = 0
    for (batch,) in loader:
        batch = batch.to(device)
        reconstruction, mu, log_variance = model(batch, stochastic=False)
        reconstruction_loss, kl_loss = loss_components(
            reconstruction, batch, mu, log_variance
        )
        batch_count = batch.shape[0]
        reconstruction_sum += float(reconstruction_loss) * batch_count
        kl_sum += float(kl_loss) * batch_count
        sample_count += batch_count
    return reconstruction_sum / sample_count, kl_sum / sample_count


def pca_reconstruction_errors(
    normalized_momenta: np.ndarray,
    train_indices: np.ndarray,
    split_indices: dict[str, np.ndarray],
    components: int,
) -> dict[str, float]:
    flattened = normalized_momenta.reshape(normalized_momenta.shape[0], -1)
    train = flattened[train_indices]
    center = train.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(train - center, full_matrices=False)
    basis = right_vectors[: min(components, right_vectors.shape[0])]
    results: dict[str, float] = {}
    for split_name, indices in split_indices.items():
        centered = flattened[indices] - center
        reconstruction = center + (centered @ basis.T) @ basis
        results[split_name] = float(
            np.sqrt(np.mean((reconstruction - flattened[indices]) ** 2))
        )
    return results


def save_text_momentum(path: Path, momentum: np.ndarray) -> None:
    np.savetxt(path, momentum, fmt="%.10e")


def main() -> int:
    args = parse_args()
    if args.latent_dim < 1:
        raise ValueError("--latent-dim must be positive")
    if args.hidden_dim < 16:
        raise ValueError("--hidden-dim must be at least 16")
    if args.generated_samples < 0:
        raise ValueError("--generated-samples cannot be negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    with np.load(args.dataset, allow_pickle=False) as dataset:
        required = {"subject_ids", "control_points", "momenta"}
        missing = required.difference(dataset.files)
        if missing:
            raise ValueError(
                f"{args.dataset} is missing required arrays: {sorted(missing)}"
            )
        subject_ids = [str(value) for value in dataset["subject_ids"].tolist()]
        control_points = np.asarray(dataset["control_points"], dtype=np.float64)
        momenta = np.asarray(dataset["momenta"], dtype=np.float64)
        deformation_kernel_width = None
        if "deformation_kernel_width" in dataset.files:
            deformation_kernel_width = float(
                np.asarray(dataset["deformation_kernel_width"]).reshape(-1)[0]
            )

    if control_points.ndim != 2 or control_points.shape[1] != 3:
        raise ValueError(f"Control points have invalid shape {control_points.shape}")
    expected_momenta_shape = (len(subject_ids), control_points.shape[0], 3)
    if momenta.shape != expected_momenta_shape:
        raise ValueError(
            f"Momenta have shape {momenta.shape}; expected {expected_momenta_shape}"
        )
    if not np.isfinite(control_points).all() or not np.isfinite(momenta).all():
        raise ValueError("Control points or momenta contain NaN/Inf")

    subject_count = len(subject_ids)
    if subject_count < 10:
        raise ValueError("At least 10 subjects are required for this training scaffold")
    if args.latent_dim >= subject_count:
        raise ValueError("Latent dimension must be smaller than subject count")

    train_indices = np.arange(subject_count, dtype=np.int64)
    split_indices = {"train": train_indices}

    coordinate_center = control_points.mean(axis=0)
    coordinate_scale = control_points.std(axis=0)
    coordinate_scale[coordinate_scale < 1e-8] = 1.0
    normalized_control_points = (
        (control_points - coordinate_center) / coordinate_scale
    ).astype(np.float32)

    mean_momentum_field = momenta[train_indices].mean(axis=0)
    residual_momenta = momenta - mean_momentum_field
    momentum_scale = np.sqrt(
        np.mean(residual_momenta[train_indices] ** 2, axis=(0, 1))
    )
    momentum_scale[momentum_scale < 1e-12] = 1.0
    normalized_momenta = (residual_momenta / momentum_scale).astype(np.float32)

    print(f"Subjects:       {subject_count}")
    print(f"Control points: {control_points.shape[0]:,}")
    print(f"Momenta tensor: {momenta.shape}")
    print(f"Fit cohort:     all {len(train_indices)} subjects (no held-out split)")
    print("Metric scope:   training reconstruction only")
    print("Building control-point kNN graph ...")
    neighbor_indices = build_knn(control_points, args.neighbors)

    device = choose_device(args.device)
    print(f"Device:         {device}")
    train_tensor = torch.from_numpy(normalized_momenta[train_indices])
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=min(args.batch_size, len(train_tensor)),
        shuffle=True,
        pin_memory=pin_memory,
    )

    model = GraphMomentaBetaVAE(
        torch.from_numpy(normalized_control_points),
        torch.from_numpy(neighbor_indices),
        args.latent_dim,
        args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_validation_mse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = args.beta * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        for (batch,) in train_loader:
            batch = batch.to(device)
            noisy_input = batch
            if args.input_noise > 0:
                noisy_input = batch + args.input_noise * torch.randn_like(batch)
            reconstruction, mu, log_variance = model(noisy_input, stochastic=True)
            reconstruction_loss, kl_loss = loss_components(
                reconstruction, batch, mu, log_variance
            )
            loss = reconstruction_loss + beta * kl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        train_mse, train_kl = evaluate(model, train_loader, device)
        # All subjects are deliberately used for fitting. This value is a
        # deterministic training reconstruction monitor, not validation loss.
        validation_mse, validation_kl = evaluate(model, train_loader, device)
        history.append(
            {
                "epoch": epoch,
                "beta": beta,
                "train_mse": train_mse,
                "train_kl": train_kl,
                "fit_mse": validation_mse,
                "fit_kl": validation_kl,
            }
        )

        if validation_mse < best_validation_mse - 1e-8:
            best_validation_mse = validation_mse
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch {epoch:4d}: train MSE={train_mse:.6f}; "
                f"fit MSE={validation_mse:.6f}; "
                f"KL={validation_kl:.6f}; beta={beta:.6g}"
            )
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    split_loaders = {"train": train_loader}
    vae_metrics: dict[str, dict[str, float]] = {}
    for split_name, loader in split_loaders.items():
        mse, kl = evaluate(model, loader, device)
        vae_metrics[split_name] = {
            "normalized_mse": mse,
            "normalized_rmse": math.sqrt(mse),
            "kl": kl,
        }

    pca_metrics = pca_reconstruction_errors(
        normalized_momenta,
        train_indices,
        split_indices,
        args.latent_dim,
    )

    all_tensor = torch.from_numpy(normalized_momenta).to(device)
    model.eval()
    with torch.no_grad():
        latent_mu, latent_log_variance = model.encode(all_tensor)
        reconstructed_normalized = model.decode(latent_mu)
    latent_mu_numpy = latent_mu.cpu().numpy()
    latent_log_variance_numpy = latent_log_variance.cpu().numpy()
    reconstructed_momenta = (
        reconstructed_normalized.cpu().numpy() * momentum_scale
        + mean_momentum_field
    )
    per_subject_rmse = np.sqrt(
        np.mean((reconstructed_momenta - momenta) ** 2, axis=(1, 2))
    )

    generated_directory = args.output_dir / "generated_momenta"
    generated_directory.mkdir(parents=True, exist_ok=True)
    generated_momenta = np.empty(
        (args.generated_samples, control_points.shape[0], 3), dtype=np.float64
    )
    if args.generated_samples:
        torch.manual_seed(args.seed + 1)
        with torch.no_grad():
            latent_samples = (
                torch.randn(args.generated_samples, args.latent_dim, device=device)
                * args.temperature
            )
            generated_normalized = model.decode(latent_samples).cpu().numpy()
        generated_momenta = (
            generated_normalized * momentum_scale + mean_momentum_field
        )
        for index, field in enumerate(generated_momenta, start=1):
            save_text_momentum(
                generated_directory / f"generated_{index:03d}_momenta.txt", field
            )

    split_by_index = np.full(subject_count, "", dtype="U10")
    for split_name, indices in split_indices.items():
        split_by_index[indices] = split_name

    checkpoint = {
        "model_state": copy.deepcopy(best_state),
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "neighbors": args.neighbors,
        "control_points": torch.from_numpy(control_points.astype(np.float32)),
        "neighbor_indices": torch.from_numpy(neighbor_indices),
        "coordinate_center": torch.from_numpy(coordinate_center.astype(np.float32)),
        "coordinate_scale": torch.from_numpy(coordinate_scale.astype(np.float32)),
        "mean_momentum_field": torch.from_numpy(mean_momentum_field.astype(np.float32)),
        "momentum_scale": torch.from_numpy(momentum_scale.astype(np.float32)),
        "subject_ids": subject_ids,
        "split": split_by_index.tolist(),
        "seed": args.seed,
        "deformation_kernel_width": deformation_kernel_width,
    }
    checkpoint_path = args.output_dir / "graph_momenta_beta_vae.pt"
    torch.save(checkpoint, checkpoint_path)

    np.savez_compressed(
        args.output_dir / "momenta_dataset_and_reconstructions.npz",
        control_points=control_points.astype(np.float32),
        momenta=momenta.astype(np.float32),
        reconstructed_momenta=reconstructed_momenta.astype(np.float32),
        generated_momenta=generated_momenta.astype(np.float32),
        latent_mu=latent_mu_numpy.astype(np.float32),
        latent_log_variance=latent_log_variance_numpy.astype(np.float32),
        subject_ids=np.asarray(subject_ids),
        split=split_by_index,
    )

    with (args.output_dir / "latent_codes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["subject", "split", "momentum_rmse"] + [
            f"z_{index + 1}" for index in range(args.latent_dim)
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for subject_index, subject_id in enumerate(subject_ids):
            row: dict[str, str | float] = {
                "subject": subject_id,
                "split": str(split_by_index[subject_index]),
                "momentum_rmse": float(per_subject_rmse[subject_index]),
            }
            row.update(
                {
                    f"z_{latent_index + 1}": float(
                        latent_mu_numpy[subject_index, latent_index]
                    )
                    for latent_index in range(args.latent_dim)
                }
            )
            writer.writerow(row)

    with (args.output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "warning": (
            f"All {subject_count} subjects were used for fitting. Metrics are "
            "training reconstruction metrics and do not establish generalization."
        ),
        "subjects": subject_count,
        "control_points": int(control_points.shape[0]),
        "latent_dimension": args.latent_dim,
        "split_subjects": {
            name: [subject_ids[index] for index in indices]
            for name, indices in split_indices.items()
        },
        "vae": vae_metrics,
        "pca_normalized_rmse_same_latent_dimension": pca_metrics,
        "best_fit_normalized_mse": best_validation_mse,
        "epochs_completed": len(history),
        "source_dataset": str(args.dataset),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    print("\nTraining complete")
    print(f"Best fit normalized MSE: {best_validation_mse:.6f}")
    print(
        "Training reconstruction normalized RMSE: "
        f"VAE={vae_metrics['train']['normalized_rmse']:.6f}; "
        f"PCA={pca_metrics['train']:.6f}"
    )
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Metrics:        {args.output_dir / 'metrics.json'}")
    print(f"Latent codes:   {args.output_dir / 'latent_codes.csv'}")
    print(f"Generated:      {generated_directory}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        OSError,
        ET.ParseError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
