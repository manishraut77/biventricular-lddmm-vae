#!/usr/bin/env python3
"""Shoot every generated combined-LV+RV momentum field into a synthetic surface.

Inputs:
  LDDMMRegisteredTemplate16/template_subject_16.vtk
  LDDMMRegisteredTemplate16/trainingdata/momenta_dataset.npz
  MomentumVAE/Template16GraphBetaVAE/sampled_momenta/temperature_0p7/*.txt

Minimal outputs:
  SyntheticHearts/Template16VAE/temperature_0p7/generated_001_surface.vtk ...

Each Deformetrica work directory is removed after a successful export. A failed
job is retained beneath .SyntheticHearts_work for diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("LDDMMRegisteredTemplate16/template_subject_16.vtk"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("LDDMMRegisteredTemplate16/trainingdata/momenta_dataset.npz"),
    )
    parser.add_argument(
        "--momenta-dir",
        type=Path,
        default=Path("MomentumVAE/Template16GraphBetaVAE/sampled_momenta/temperature_0p7"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("SyntheticHearts/Template16VAE/temperature_0p7")
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path(".SyntheticHearts_work")
    )
    parser.add_argument("--docker-image", default="deformetrica-kcl:4.3")
    parser.add_argument("--docker-platform", default="linux/amd64")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--timepoints", type=int, default=10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated surface files with matching names.",
    )
    args = parser.parse_args()
    if args.threads < 1 or args.timepoints < 2:
        parser.error("--threads must be >=1 and --timepoints must be >=2")
    return args


def require_file(path: Path, description: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def require_inside(path: Path, root: Path, description: str) -> Path:
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description} must be inside project root: {path}") from exc
    return path


def find_generated_momenta(directory: Path) -> list[Path]:
    result = sorted(
        path.resolve()
        for path in directory.glob("generated_*_momenta.txt")
        if re.fullmatch(r"generated_\d+_momenta\.txt", path.name)
    )
    if not result:
        raise FileNotFoundError(
            f"No generated_###_momenta.txt files found in {directory.resolve()}"
        )
    return result


def model_xml(
    template_rel: str,
    control_rel: str,
    momentum_rel: str,
    kernel_width: float,
    timepoints: int,
) -> str:
    return f"""<?xml version="1.0"?>
<model>
  <model-type>Shooting</model-type>
  <dimension>3</dimension>
  <initial-control-points>{escape(control_rel)}</initial-control-points>
  <initial-momenta>{escape(momentum_rel)}</initial-momenta>
  <template>
    <object id="synthetic_biventricular_surface">
      <deformable-object-type>SurfaceMesh</deformable-object-type>
      <attachment-type>Current</attachment-type>
      <kernel-type>keops</kernel-type>
      <kernel-device>cpu</kernel-device>
      <kernel-width>{kernel_width:g}</kernel-width>
      <noise-std>1</noise-std>
      <filename>{escape(template_rel)}</filename>
    </object>
  </template>
  <deformation-parameters>
    <kernel-type>keops</kernel-type>
    <kernel-device>cpu</kernel-device>
    <kernel-width>{kernel_width:g}</kernel-width>
    <number-of-timepoints>{timepoints}</number-of-timepoints>
    <t0>0</t0>
    <tmin>0</tmin>
    <tmax>1</tmax>
  </deformation-parameters>
</model>
"""


OPTIMIZATION_XML = """<?xml version="1.0"?>
<optimization-parameters>
  <use-rk2>On</use-rk2>
  <gpu-mode>None</gpu-mode>
</optimization-parameters>
"""


def run_docker(
    project_root: Path,
    job_dir: Path,
    image: str,
    platform_name: str,
    threads: int,
) -> None:
    job_rel = job_dir.relative_to(project_root)
    command = [
        "docker", "run", "--rm",
        "--platform", platform_name,
        "--cpus", str(threads),
        "-e", f"OMP_NUM_THREADS={threads}",
        "-v", f"{project_root}:/work",
        "-w", "/work",
        image,
        "compute",
        (job_rel / "model.xml").as_posix(),
        "--parameters", (job_rel / "optimization_parameters.xml").as_posix(),
        "--output", (job_rel / "output").as_posix(),
        "--verbosity", "INFO",
    ]
    subprocess.run(command, cwd=project_root, check=True)


def final_surface(output_dir: Path) -> Path:
    candidates = sorted(
        output_dir.glob(
            "Shooting__GeodesicFlow__synthetic_biventricular_surface"
            "__tp_*__age_1.00.vtk"
        )
    )
    if not candidates:
        candidates = sorted(output_dir.glob("*__age_1.00.vtk"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one final age-1 surface in {output_dir}; found {len(candidates)}"
        )
    return candidates[0]


def read_points(path: Path) -> np.ndarray:
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        vtk = None

    if vtk is not None:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        mesh = reader.GetOutput()
        if mesh is None or mesh.GetNumberOfPoints() == 0:
            raise RuntimeError(f"Could not read VTK surface: {path}")
        return np.asarray(vtk_to_numpy(mesh.GetPoints().GetData()), dtype=np.float64)

    tokens = path.read_text(encoding="ascii", errors="strict").split()
    try:
        index = next(i for i, token in enumerate(tokens) if token.upper() == "POINTS")
        count = int(tokens[index + 1])
        start = index + 3
        values = np.asarray(tokens[start : start + 3 * count], dtype=np.float64)
    except (StopIteration, ValueError, IndexError) as exc:
        raise RuntimeError(
            "VTK Python is unavailable and the file is not readable ASCII legacy VTK: "
            f"{path}"
        ) from exc
    if values.size != 3 * count:
        raise RuntimeError(f"Incomplete POINTS section in {path}")
    return values.reshape(count, 3)


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    template = require_inside(require_file(args.template, "average template"), root, "Template")
    dataset_path = require_inside(require_file(args.dataset, "momentum dataset"), root, "Dataset")
    momenta_dir = require_inside(args.momenta_dir.resolve(), root, "Momenta directory")
    generated_files = find_generated_momenta(momenta_dir)

    with np.load(dataset_path, allow_pickle=False) as dataset:
        for key in ("control_points", "momenta", "subject_ids"):
            if key not in dataset.files:
                raise ValueError(f"Dataset is missing required array: {key}")
        control_points = np.asarray(dataset["control_points"], dtype=np.float64)
        training_momenta = np.asarray(dataset["momenta"], dtype=np.float64)
        subject_ids = [str(value) for value in dataset["subject_ids"].tolist()]
        if "deformation_kernel_width" not in dataset.files:
            raise ValueError("Dataset does not contain deformation_kernel_width")
        kernel_width = float(
            np.asarray(dataset["deformation_kernel_width"]).reshape(-1)[0]
        )

    expected = (control_points.shape[0], 3)
    if control_points.shape != expected:
        raise ValueError(f"Control points must have shape (N,3); got {control_points.shape}")
    if training_momenta.shape != (len(subject_ids), *expected):
        raise ValueError(
            f"Training momenta have unexpected shape {training_momenta.shape}"
        )
    if not np.isfinite(control_points).all() or not np.isfinite(training_momenta).all():
        raise ValueError("Training dataset contains NaN or Inf")
    if not np.isfinite(kernel_width) or kernel_width <= 0:
        raise ValueError(f"Invalid deformation kernel width: {kernel_width}")

    output_dir = require_inside(args.output_dir.resolve(), root, "Output directory")
    work_root = require_inside(args.work_dir.resolve(), root, "Work directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    control_file = work_root / "control_points.txt"
    np.savetxt(control_file, control_points, fmt="%.10e")
    template_points = read_points(template)

    print(f"Generated fields:       {len(generated_files)}")
    print(f"Template points:        {len(template_points):,}")
    print(f"Control points:         {len(control_points):,}")
    print(f"Deformation kernel:     {kernel_width:g} mm")
    print(f"Final surfaces:         {output_dir}")
    print("These are combined LV+RV surfaces, not tetrahedral volume meshes.\n")

    rows: list[dict[str, object]] = []
    generated_flat: list[np.ndarray] = []
    training_flat = training_momenta.reshape(len(training_momenta), -1)

    for sequence, momentum_path in enumerate(generated_files, start=1):
        match = re.search(r"generated_(\d+)_momenta", momentum_path.stem)
        assert match is not None
        sample_id = match.group(1)
        output_path = output_dir / f"generated_{sample_id}_surface.vtk"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite deliberately."
            )

        momentum = np.atleast_2d(np.loadtxt(momentum_path, dtype=np.float64))
        if momentum.shape != expected:
            raise ValueError(
                f"{momentum_path} has shape {momentum.shape}; expected {expected}"
            )
        if not np.isfinite(momentum).all():
            raise ValueError(f"Generated momentum contains NaN/Inf: {momentum_path}")

        job_dir = work_root / f"generated_{sample_id}"
        if job_dir.exists():
            raise FileExistsError(
                f"Work directory already exists: {job_dir}. Inspect it before retrying."
            )
        (job_dir / "output").mkdir(parents=True)
        momentum_copy = job_dir / "momenta.txt"
        np.savetxt(momentum_copy, momentum, fmt="%.10e")

        rel = lambda p: Path(os.path.relpath(p, job_dir)).as_posix()
        (job_dir / "model.xml").write_text(
            model_xml(
                rel(template),
                rel(control_file),
                rel(momentum_copy),
                kernel_width,
                args.timepoints,
            ),
            encoding="utf-8",
        )
        (job_dir / "optimization_parameters.xml").write_text(
            OPTIMIZATION_XML, encoding="utf-8"
        )

        print(f"[{sequence:02d}/{len(generated_files):02d}] generated_{sample_id}")
        run_docker(
            root, job_dir, args.docker_image, args.docker_platform, args.threads
        )
        shot = final_surface(job_dir / "output")
        if output_path.exists():
            output_path.unlink()
        shutil.copy2(shot, output_path)

        shot_points = read_points(output_path)
        if shot_points.shape != template_points.shape:
            raise RuntimeError(
                f"Point count changed during shooting for generated_{sample_id}: "
                f"{shot_points.shape} vs {template_points.shape}"
            )
        displacement = np.linalg.norm(shot_points - template_points, axis=1)
        momentum_flat = momentum.reshape(-1)
        generated_flat.append(momentum_flat)
        train_rmse = np.sqrt(
            np.mean((training_flat - momentum_flat[None, :]) ** 2, axis=1)
        )
        nearest_index = int(np.argmin(train_rmse))

        rows.append(
            {
                "sample": f"generated_{sample_id}",
                "surface": output_path.as_posix(),
                "finite_points": bool(np.isfinite(shot_points).all()),
                "mean_displacement_mm": float(displacement.mean()),
                "p95_displacement_mm": float(np.percentile(displacement, 95)),
                "max_displacement_mm": float(displacement.max()),
                "nearest_training_subject": subject_ids[nearest_index],
                "nearest_training_momentum_rmse": float(train_rmse[nearest_index]),
            }
        )
        print(
            f"  displacement mean={displacement.mean():.3f} mm; "
            f"P95={np.percentile(displacement, 95):.3f}; "
            f"nearest training={subject_ids[nearest_index]}"
        )
        shutil.rmtree(job_dir)

    generated_matrix = np.stack(generated_flat)
    if len(generated_matrix) > 1:
        pairwise = np.sqrt(
            np.mean(
                (generated_matrix[:, None, :] - generated_matrix[None, :, :]) ** 2,
                axis=2,
            )
        )
        upper = pairwise[np.triu_indices(len(pairwise), k=1)]
        diversity = {
            "pairwise_momentum_rmse_min": float(upper.min()),
            "pairwise_momentum_rmse_median": float(np.median(upper)),
            "pairwise_momentum_rmse_max": float(upper.max()),
        }
    else:
        diversity = {}

    report_path = output_dir / "generation_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "model": "Template16GraphBetaVAE",
        "source_template": template.relative_to(root).as_posix(),
        "source_dataset": dataset_path.relative_to(root).as_posix(),
        "source_momenta_directory": momenta_dir.relative_to(root).as_posix(),
        "deformation_kernel_width_mm": kernel_width,
        "control_points": int(len(control_points)),
        "samples": len(rows),
        "output_kind": "combined LV+RV surface meshes; not tetrahedral hearts",
        **diversity,
    }
    metadata_path = output_dir / "generation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    try:
        control_file.unlink()
        work_root.rmdir()
    except OSError:
        pass

    print("\nSynthetic surface generation complete")
    print(f"Surfaces: {output_dir}")
    print(f"Report:   {report_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

