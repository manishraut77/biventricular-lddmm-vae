#!/usr/bin/env python3
"""Register 20 combined biventricular surfaces to fixed subject 16.

Each ``LDDMMSurfacesTemplate16/NN_surface.vtk`` file is registered as ONE SurfaceMesh
observation with a de Rham-current attachment. Subject 16 is the fixed template
(``freeze-template=On``); only subject momenta are estimated.

The script runs Deformetrica in the existing ``deformetrica-kcl:4.3`` Docker
image, validates and exports the final products, then removes the large flow,
state, log and KeOps-cache files that Deformetrica creates.  If registration or
validation fails, the work directory is deliberately retained for diagnosis.

Default final layout::

    LDDMMRegisteredTemplate16/
      template_subject_16.vtk
      01_registered.vtk ... 20_registered.vtk
      registration_metadata.json
      registration_quality.csv              # when VTK is installed
      trainingdata/
        control_points.txt                   # Deformetrica-native format
        momenta.txt                          # Deformetrica-native format
        momenta_dataset.npz                  # ready for PCA/VAE input
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


OBJECT_ID = "combined_biventricular_surface"
DEFAULT_SURFACE_DIR = Path("LDDMMSurfacesTemplate16")
DEFAULT_OUTPUT_DIR = Path("LDDMMRegisteredTemplate16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface-dir",
        type=Path,
        default=DEFAULT_SURFACE_DIR,
        help=f"combined input surfaces (default: {DEFAULT_SURFACE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"minimal final output folder (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--initial-template",
        default="16",
        help="fixed template subject (default: 16)",
    )
    parser.add_argument("--current-width", type=float, default=6.0)
    parser.add_argument("--noise-std", type=float, default=5.0)
    parser.add_argument("--deformation-width", type=float, default=10.0)
    parser.add_argument("--control-point-spacing", type=float, default=10.0)
    parser.add_argument("--timepoints", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--initial-step-size", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument(
        "--docker-image", default="deformetrica-kcl:4.3"
    )
    parser.add_argument(
        "--docker-platform", default="linux/amd64"
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="skip point-to-triangle reconstruction distances",
    )
    args = parser.parse_args()

    positive = {
        "--current-width": args.current_width,
        "--noise-std": args.noise_std,
        "--deformation-width": args.deformation_width,
        "--control-point-spacing": args.control_point_spacing,
        "--timepoints": args.timepoints,
        "--max-iterations": args.max_iterations,
        "--initial-step-size": args.initial_step_size,
        "--threads": args.threads,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if not re.fullmatch(r"\d{2}", args.initial_template):
        parser.error("--initial-template must be a two-digit subject ID")
    if args.initial_template != "16":
        parser.error("this pipeline is intentionally fixed to subject 16")
    return args


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def relative_xml_path(target: Path, xml_directory: Path) -> str:
    return Path(os.path.relpath(target, xml_directory)).as_posix()


def validate_surface_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing surface: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty surface: {path}")
    with path.open("rb") as stream:
        header = stream.read(8192).upper()
    if b"POLYDATA" not in header:
        raise ValueError(f"Expected legacy VTK POLYDATA: {path}")


def build_model_xml(args: argparse.Namespace, template_path: str) -> str:
    return f'''<?xml version="1.0"?>
<model>
    <model-type>DeterministicAtlas</model-type>
    <dimension>3</dimension>
    <random-seed>42</random-seed>
    <initial-cp-spacing>{args.control_point_spacing:g}</initial-cp-spacing>

    <template>
        <object id="{OBJECT_ID}">
            <deformable-object-type>SurfaceMesh</deformable-object-type>
            <attachment-type>Current</attachment-type>
            <noise-std>{args.noise_std:g}</noise-std>
            <kernel-type>keops</kernel-type>
            <kernel-device>cpu</kernel-device>
            <kernel-width>{args.current_width:g}</kernel-width>
            <filename>{template_path}</filename>
        </object>
    </template>

    <deformation-parameters>
        <kernel-width>{args.deformation_width:g}</kernel-width>
        <kernel-type>keops</kernel-type>
        <kernel-device>cpu</kernel-device>
        <number-of-timepoints>{args.timepoints}</number-of-timepoints>
    </deformation-parameters>
</model>
'''


def build_dataset_xml(cases: list[str], paths: dict[str, str]) -> str:
    lines = ['<?xml version="1.0"?>', "<data-set>"]
    for case in cases:
        lines.extend(
            [
                f'    <subject id="{case}">',
                '        <visit id="baseline">',
                f'            <filename object_id="{OBJECT_ID}">'
                f"{paths[case]}</filename>",
                "        </visit>",
                "    </subject>",
            ]
        )
    lines.append("</data-set>")
    return "\n".join(lines) + "\n"


def build_optimization_xml(args: argparse.Namespace) -> str:
    # Saving only at the final requested iteration avoids repeated checkpoints.
    return f'''<?xml version="1.0"?>
<optimization-parameters>
    <optimization-method-type>GradientAscent</optimization-method-type>
    <initial-step-size>{args.initial_step_size:g}</initial-step-size>
    <max-iterations>{args.max_iterations}</max-iterations>
    <convergence-tolerance>1e-4</convergence-tolerance>
    <freeze-template>On</freeze-template>
    <freeze-control-points>On</freeze-control-points>
    <number-of-processes>1</number-of-processes>
    <print-every-n-iters>5</print-every-n-iters>
    <save-every-n-iters>{args.max_iterations}</save-every-n-iters>
</optimization-parameters>
'''


def write_configuration(
    work_dir: Path,
    args: argparse.Namespace,
    cases: list[str],
    surfaces: dict[str, Path],
) -> None:
    xml_paths = {
        case: relative_xml_path(path, work_dir) for case, path in surfaces.items()
    }
    (work_dir / "model.xml").write_text(
        build_model_xml(args, xml_paths[args.initial_template]), encoding="utf-8"
    )
    (work_dir / "data_set.xml").write_text(
        build_dataset_xml(cases, xml_paths), encoding="utf-8"
    )
    (work_dir / "optimization_parameters.xml").write_text(
        build_optimization_xml(args), encoding="utf-8"
    )


def check_docker(image: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is not installed or is not on PATH")
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Docker image {image!r} is unavailable, or Docker Desktop is not running"
        )


def run_deformetrica(
    project: Path, work_dir: Path, args: argparse.Namespace
) -> None:
    work_relative = work_dir.relative_to(project).as_posix()
    cache_dir = work_dir / "keops_cache"
    output_dir = work_dir / "output"
    cache_dir.mkdir()
    output_dir.mkdir()

    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        args.docker_platform,
        "-e",
        f"OMP_NUM_THREADS={args.threads}",
        "-v",
        f"{project}:/work",
        "-v",
        f"{cache_dir}:/root/.cache",
        "-w",
        "/work",
        args.docker_image,
        "estimate",
        f"{work_relative}/model.xml",
        f"{work_relative}/data_set.xml",
        "--parameters",
        f"{work_relative}/optimization_parameters.xml",
        "--output",
        f"{work_relative}/output",
        "--verbosity",
        "INFO",
    ]
    print("\nStarting Deformetrica. This is the long-running step.\n", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Deformetrica exited with status {completed.returncode}"
        )


def find_one(directory: Path, pattern: str, description: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches[:8]) or "none"
        raise FileNotFoundError(
            f"Expected exactly one {description} matching {pattern!r} in "
            f"{directory}; found {len(matches)} ({names})"
        )
    return matches[0]


def load_control_points(path: Path) -> np.ndarray:
    values = np.atleast_2d(np.loadtxt(path, dtype=np.float64, comments="#"))
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError(f"Invalid control-point array shape {values.shape}: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Control points contain NaN or infinity: {path}")
    return values


def load_momenta(
    path: Path, subject_count: int, control_point_count: int
) -> np.ndarray:
    nonempty = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not nonempty:
        raise ValueError(f"Empty momenta file: {path}")

    first = nonempty[0].split()
    has_integer_header = (
        len(first) == 3 and all(re.fullmatch(r"[+-]?\d+", token) for token in first)
    )
    if has_integer_header:
        n_subjects, n_points, dimension = map(int, first)
        if (n_subjects, n_points, dimension) != (
            subject_count,
            control_point_count,
            3,
        ):
            raise ValueError(
                "Momenta header is "
                f"{(n_subjects, n_points, dimension)}; expected "
                f"{(subject_count, control_point_count, 3)}"
            )
        numeric_text = "\n".join(nonempty[1:])
        values = np.atleast_2d(
            np.loadtxt(io.StringIO(numeric_text), dtype=np.float64)
        )
    else:
        values = np.atleast_2d(np.loadtxt(path, dtype=np.float64, comments="#"))

    expected = subject_count * control_point_count
    if values.shape != (expected, 3):
        raise ValueError(
            f"Momenta values have shape {values.shape}; expected {(expected, 3)}"
        )
    momenta = values.reshape(subject_count, control_point_count, 3)
    if not np.isfinite(momenta).all():
        raise ValueError(f"Momenta contain NaN or infinity: {path}")
    return momenta


def load_polydata(path: Path):
    import vtk

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(reader.GetOutput())
    if result.GetNumberOfPoints() == 0 or result.GetNumberOfPolys() == 0:
        raise ValueError(f"VTK surface is empty or invalid: {path}")
    return result


def directed_surface_distances(source, target) -> np.ndarray:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    evaluator = vtk.vtkImplicitPolyDataDistance()
    evaluator.SetInput(target)
    points = vtk_to_numpy(source.GetPoints().GetData())
    distances = np.fromiter(
        (abs(evaluator.EvaluateFunction(point)) for point in points),
        dtype=np.float64,
        count=len(points),
    )
    if not np.isfinite(distances).all():
        raise ValueError("Surface-distance calculation returned NaN or infinity")
    return distances


def summarize_distances(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_mm": float(np.mean(values)),
        f"{prefix}_median_mm": float(np.median(values)),
        f"{prefix}_rms_mm": float(math.sqrt(np.mean(values * values))),
        f"{prefix}_p95_mm": float(np.percentile(values, 95)),
        f"{prefix}_max_mm": float(np.max(values)),
    }


def calculate_quality(
    cases: list[str],
    targets: dict[str, Path],
    registered: dict[str, Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        target = load_polydata(targets[case])
        reconstruction = load_polydata(registered[case])
        target_to_reconstruction = directed_surface_distances(
            target, reconstruction
        )
        reconstruction_to_target = directed_surface_distances(
            reconstruction, target
        )
        symmetric = np.concatenate(
            [target_to_reconstruction, reconstruction_to_target]
        )
        row: dict[str, object] = {
            "subject_id": case,
            "target_points": target.GetNumberOfPoints(),
            "registered_points": reconstruction.GetNumberOfPoints(),
        }
        row.update(summarize_distances(target_to_reconstruction, "target_to_registered"))
        row.update(summarize_distances(reconstruction_to_target, "registered_to_target"))
        row.update(summarize_distances(symmetric, "symmetric"))
        rows.append(row)
        print(
            f"  subject {case}: symmetric mean={row['symmetric_mean_mm']:.3f} mm; "
            f"RMS={row['symmetric_rms_mm']:.3f} mm; "
            f"P95={row['symmetric_p95_mm']:.3f} mm; "
            f"max={row['symmetric_max_mm']:.3f} mm"
        )
    return rows


def write_quality_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_results(
    project: Path,
    work_dir: Path,
    staging: Path,
    args: argparse.Namespace,
    cases: list[str],
    surfaces: dict[str, Path],
) -> tuple[int, tuple[int, int, int], str]:
    raw_output = work_dir / "output"
    control_source = find_one(
        raw_output,
        "*EstimatedParameters*ControlPoints*.txt",
        "estimated control-point file",
    )
    momenta_source = find_one(
        raw_output,
        "*EstimatedParameters*Momenta*.txt",
        "estimated momenta file",
    )
    # With freeze-template=On, the authoritative template is the original
    # subject-16 surface rather than an estimated population-average surface.
    template_source = surfaces[args.initial_template]

    reconstruction_sources = {
        case: find_one(
            raw_output,
            f"*Reconstruction*subject_{case}.vtk",
            f"reconstruction for subject {case}",
        )
        for case in cases
    }
    validate_surface_file(template_source)
    for path in reconstruction_sources.values():
        validate_surface_file(path)

    control_points = load_control_points(control_source)
    momenta = load_momenta(
        momenta_source, len(cases), control_points.shape[0]
    )

    staging.mkdir()
    training = staging / "trainingdata"
    training.mkdir()
    shutil.copy2(template_source, staging / "template_subject_16.vtk")
    registered: dict[str, Path] = {}
    for case, source in reconstruction_sources.items():
        destination = staging / f"{case}_registered.vtk"
        shutil.copy2(source, destination)
        registered[case] = destination

    shutil.copy2(control_source, training / "control_points.txt")
    shutil.copy2(momenta_source, training / "momenta.txt")
    np.savez_compressed(
        training / "momenta_dataset.npz",
        subject_ids=np.asarray(cases, dtype="U2"),
        control_points=control_points.astype(np.float32),
        momenta=momenta.astype(np.float32),
        deformation_kernel_width=np.float32(args.deformation_width),
        current_kernel_width=np.float32(args.current_width),
        control_point_spacing=np.float32(args.control_point_spacing),
        initial_template_subject=np.asarray(args.initial_template),
        template_was_optimized=np.asarray(False),
    )

    qa_status = "skipped by --skip-qa"
    qa_rows: list[dict[str, object]] = []
    if not args.skip_qa:
        try:
            import vtk  # noqa: F401

            print("\nPoint-to-triangle reconstruction QA:")
            qa_rows = calculate_quality(cases, surfaces, registered)
            write_quality_csv(staging / "registration_quality.csv", qa_rows)
            qa_status = "completed"
        except ImportError:
            qa_status = "skipped because the Python VTK package is unavailable"
            print(f"WARNING: {qa_status}", file=sys.stderr)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "fixed-template deterministic atlas registration",
        "object_id": OBJECT_ID,
        "observation_objects_per_subject": 1,
        "input_representation": "one combined LV+RV surface per subject",
        "attachment_type": "Current",
        "initial_template_subject": args.initial_template,
        "template_was_optimized": False,
        "template_interpretation": (
            "fixed geometry of subject 16; not a population-average template"
        ),
        "subjects": cases,
        "subject_count": len(cases),
        "control_point_count": int(control_points.shape[0]),
        "momenta_shape": list(momenta.shape),
        "parameters": {
            "current_kernel_width": args.current_width,
            "noise_std": args.noise_std,
            "deformation_kernel_width": args.deformation_width,
            "control_point_spacing": args.control_point_spacing,
            "number_of_timepoints": args.timepoints,
            "maximum_iterations": args.max_iterations,
            "initial_step_size": args.initial_step_size,
            "freeze_template": True,
            "freeze_control_points": True,
            "random_seed": 42,
        },
        "software": {
            "docker_image": args.docker_image,
            "docker_platform": args.docker_platform,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "qa_status": qa_status,
        "qa_subject_count": len(qa_rows),
        "training_note": (
            "Split subjects before normalization; fit all normalization statistics "
            "on training subjects only."
        ),
    }
    (staging / "registration_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    # Read the NPZ back and validate the exact persisted training tensors.
    with np.load(training / "momenta_dataset.npz") as dataset:
        if dataset["control_points"].shape != control_points.shape:
            raise ValueError("Persisted control-point tensor has the wrong shape")
        if dataset["momenta"].shape != momenta.shape:
            raise ValueError("Persisted momenta tensor has the wrong shape")
        if not np.isfinite(dataset["momenta"]).all():
            raise ValueError("Persisted momenta contain NaN or infinity")

    return control_points.shape[0], momenta.shape, qa_status


def main() -> int:
    args = parse_args()
    project = Path.cwd().resolve()
    surface_dir = (project / args.surface_dir).resolve()
    final_dir = (project / args.output_dir).resolve()

    if not is_inside(surface_dir, project):
        raise ValueError("--surface-dir must be inside the repository")
    if not is_inside(final_dir, project) or final_dir == project:
        raise ValueError("--output-dir must be a folder inside the repository")
    if surface_dir == final_dir:
        raise ValueError("Input and output directories cannot be the same")
    if not surface_dir.is_dir():
        raise FileNotFoundError(f"Surface directory does not exist: {surface_dir}")

    cases = [f"{number:02d}" for number in range(1, 21)]
    if args.initial_template not in cases:
        raise ValueError("--initial-template must be included in the subject cohort")
    surfaces = {case: surface_dir / f"{case}_surface.vtk" for case in cases}
    for path in surfaces.values():
        validate_surface_file(path)

    work_dir = final_dir.parent / f".{final_dir.name}_work"
    staging = final_dir.parent / f".{final_dir.name}.staging"
    for path, label in (
        (final_dir, "final output"),
        (work_dir, "temporary registration work"),
        (staging, "temporary export staging"),
    ):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing {label}: {path}\n"
                "Move or rename it, inspect it first, then run again."
            )

    check_docker(args.docker_image)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir()
    try:
        write_configuration(work_dir, args, cases, surfaces)
        print(f"Validated surfaces:      {len(cases)} ({cases[0]}--{cases[-1]})")
        print("Registration object:     one combined LV+RV current per subject")
        print(f"Fixed template:          subject {args.initial_template}")
        print("Template optimization:   disabled (freeze-template=On)")
        print(f"Current kernel:          {args.current_width:g}")
        print(f"Observation noise std:   {args.noise_std:g}")
        print(f"Deformation kernel:      {args.deformation_width:g} mm")
        print(f"Control-point spacing:   {args.control_point_spacing:g} mm")
        print(f"Maximum iterations:      {args.max_iterations}")
        print(f"Final destination:       {final_dir.relative_to(project)}")
        if sys.platform == "darwin":
            print("macOS reminder: keep the Mac awake until this command finishes.")

        run_deformetrica(project, work_dir, args)
        print("\nRegistration finished; validating and exporting final products.")
        control_point_count, momenta_shape, qa_status = export_results(
            project, work_dir, staging, args, cases, surfaces
        )
        staging.rename(final_dir)

        try:
            shutil.rmtree(work_dir)
            cleanup_status = "removed"
        except OSError as exc:
            cleanup_status = f"not removed ({exc})"
            print(
                f"WARNING: final output is valid, but temporary work remains: {work_dir}",
                file=sys.stderr,
            )

        print("\nRegistration and export complete")
        print(f"Fixed template:          {final_dir / 'template_subject_16.vtk'}")
        print(f"Registered surfaces:     {len(cases)} in {final_dir}")
        print(f"Control points:          {control_point_count:,}")
        print(f"Training momenta shape:  {momenta_shape}")
        print(f"Training dataset:        {final_dir / 'trainingdata/momenta_dataset.npz'}")
        print(f"Reconstruction QA:       {qa_status}")
        print(f"Temporary Deformetrica:  {cleanup_status}")
        return 0
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        print(
            f"\nRegistration did not publish a final result. Diagnostic files were "
            f"retained in:\n  {work_dir}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
