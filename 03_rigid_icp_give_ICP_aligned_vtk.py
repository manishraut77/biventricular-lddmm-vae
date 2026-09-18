#!/usr/bin/env python3
"""Rigidly align a folder of tetrahedral VTK hearts to one template.

Default folder layout (run this script from the project directory):

    ./AlignedVtkData/01.vtk ... 20.vtk
    ./ICPAlignedVtkTemplate16/   # created by this script

Only a cleaned, decimated surface copy is used to estimate each ICP transform.
The resulting rotation and translation are then applied to every point in the
original full tetrahedral mesh. Connectivity and data arrays are preserved.

Install the only dependency with:

    python -m pip install vtk

Run with:

    python rigid_icp.py

Use ``--overwrite`` when deliberately replacing a previous run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import vtk
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Python package 'vtk' is required. Install it with:\n"
        "    python -m pip install vtk"
    ) from exc


def natural_key(path: Path) -> list[object]:
    """Sort 2.vtk before 10.vtk while also supporting names such as 01.vtk."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def read_unstructured_grid(path: Path) -> Any:
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(path))

    # Preserve all legacy VTK point/cell arrays, not only the active arrays.
    for method_name in (
        "ReadAllScalarsOn",
        "ReadAllVectorsOn",
        "ReadAllNormalsOn",
        "ReadAllTensorsOn",
        "ReadAllColorScalarsOn",
        "ReadAllFieldsOn",
    ):
        method = getattr(reader, method_name, None)
        if method is not None:
            method()

    reader.Update()
    output = reader.GetOutput()
    if output is None or output.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Could not read an unstructured grid from {path}")

    mesh = vtk.vtkUnstructuredGrid()
    mesh.DeepCopy(output)
    return mesh


def clean_polydata(polydata: Any) -> Any:
    """Merge coincident points and discard non-triangle cells."""
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(polydata)
    triangles.PassVertsOff()
    triangles.PassLinesOff()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.PointMergingOn()
    # Exact duplicates are merged without collapsing nearby anatomical points.
    clean.SetTolerance(0.0)
    if hasattr(clean, "ConvertPolysToLinesOff"):
        clean.ConvertPolysToLinesOff()
    if hasattr(clean, "ConvertLinesToPointsOff"):
        clean.ConvertLinesToPointsOff()
    clean.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(clean.GetOutput())
    return result


def make_registration_surface(mesh: Any, maximum_triangles: int) -> Any:
    """Extract and decimate a temporary boundary surface for ICP."""
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(mesh)
    surface_filter.Update()

    surface = clean_polydata(surface_filter.GetOutput())
    number_of_triangles = surface.GetNumberOfCells()
    if number_of_triangles == 0 or surface.GetNumberOfPoints() < 3:
        raise RuntimeError("Surface extraction produced an empty surface")

    if maximum_triangles > 0 and number_of_triangles > maximum_triangles:
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputData(surface)
        target_reduction = 1.0 - maximum_triangles / number_of_triangles
        decimator.SetTargetReduction(target_reduction)
        if hasattr(decimator, "VolumePreservationOn"):
            decimator.VolumePreservationOn()
        decimator.Update()
        surface = clean_polydata(decimator.GetOutput())

    return surface


def rigid_icp(
    source_surface: Any,
    template_surface: Any,
    maximum_iterations: int,
    maximum_landmarks: int,
    convergence_tolerance: float,
) -> Any:
    """Estimate a source-to-template transform with no scale or deformation."""
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_surface)
    icp.SetTarget(template_surface)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.StartByMatchingCentroidsOn()
    icp.SetMaximumNumberOfIterations(maximum_iterations)
    icp.SetMaximumNumberOfLandmarks(maximum_landmarks)
    icp.CheckMeanDistanceOn()
    icp.SetMeanDistanceModeToRMS()
    icp.SetMaximumMeanDistance(convergence_tolerance)
    icp.Modified()
    icp.Update()
    return icp


def transform_polydata(polydata: Any, transform: Any) -> Any:
    # vtkTransformFilter supports vtkPolyData and avoids the deprecated
    # vtkTransformPolyDataFilter in VTK 9.7+.
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(transform_filter.GetOutput())
    return result


def transform_full_mesh(mesh: Any, transform: Any) -> Any:
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetInputData(mesh)
    transform_filter.SetTransform(transform)
    # Rotate every three-component vector array, including cardiac fibres.
    if hasattr(transform_filter, "TransformAllInputVectorsOn"):
        transform_filter.TransformAllInputVectorsOn()
    transform_filter.Update()

    result = vtk.vtkUnstructuredGrid()
    result.DeepCopy(transform_filter.GetOutput())
    return result


def write_unstructured_grid(mesh: Any, path: Path) -> None:
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(mesh)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"VTK failed to write {path}")


def matrix_as_rows(transform: Any) -> list[list[float]]:
    matrix = transform.GetMatrix()
    return [
        [float(matrix.GetElement(row, column)) for column in range(4)]
        for row in range(4)
    ]


def _directed_nearest_distances(source: Any, target: Any) -> list[float]:
    locator_class = getattr(vtk, "vtkStaticPointLocator", vtk.vtkPointLocator)
    locator = locator_class()
    locator.SetDataSet(target)
    locator.BuildLocator()

    distances: list[float] = []
    for point_id in range(source.GetNumberOfPoints()):
        point = source.GetPoint(point_id)
        nearest_id = locator.FindClosestPoint(point)
        nearest = target.GetPoint(nearest_id)
        squared = sum((point[axis] - nearest[axis]) ** 2 for axis in range(3))
        distances.append(math.sqrt(squared))
    return distances


def symmetric_nearest_neighbor_summary(first: Any, second: Any) -> dict[str, float]:
    """Approximate bidirectional surface mismatch for before/after QA."""
    distances = (
        _directed_nearest_distances(first, second)
        + _directed_nearest_distances(second, first)
    )
    distances.sort()
    count = len(distances)
    percentile_index = min(count - 1, math.ceil(0.95 * count) - 1)
    return {
        "mean": float(sum(distances) / count),
        "rms": float(math.sqrt(sum(value * value for value in distances) / count)),
        "p95": float(distances[percentile_index]),
        "maximum": float(distances[-1]),
    }


def identity_matrix() -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(4)]
            for row in range(4)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rigidly align all VTK hearts to one fixed template."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("AlignedVtkData"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ICPAlignedVtkTemplate16")
    )
    parser.add_argument(
        "--template",
        default="16.vtk",
        help="Template filename inside --input-dir (default: 16.vtk)",
    )
    parser.add_argument(
        "--surface-triangles",
        type=int,
        default=30_000,
        help="Maximum triangles in each temporary ICP surface (default: 30000)",
    )
    parser.add_argument("--landmarks", type=int, default=30_000)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="ICP transform convergence tolerance in mesh coordinate units",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs from a previous run",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if Path(args.template).name != args.template:
        raise ValueError("--template must be a filename, not a path")
    if args.surface_triangles < 100:
        raise ValueError("--surface-triangles must be at least 100")
    if args.landmarks < 3:
        raise ValueError("--landmarks must be at least 3")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)

    vtk_files = sorted(
        (path for path in args.input_dir.iterdir()
         if path.is_file() and path.suffix.lower() == ".vtk"),
        key=natural_key,
    )
    if not vtk_files:
        raise FileNotFoundError(f"No .vtk files found in {args.input_dir}")

    template_path = args.input_dir / args.template
    if template_path not in vtk_files:
        raise FileNotFoundError(
            f"Template {args.template!r} was not found in {args.input_dir}. "
            "If the file is literally named o1.vtk, pass --template o1.vtk."
        )

    report_path = args.output_dir / "icp_transforms.json"
    planned_outputs = [args.output_dir / path.name for path in vtk_files]
    collisions = [path for path in [*planned_outputs, report_path] if path.exists()]
    if collisions and not args.overwrite:
        preview = ", ".join(str(path) for path in collisions[:3])
        raise FileExistsError(
            f"Output already exists ({preview}). Use --overwrite to replace a "
            "previous run."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(vtk_files)} VTK hearts")
    print(f"Fixed template: {template_path}")

    print("Preparing template registration surface ...")
    template_mesh = read_unstructured_grid(template_path)
    template_surface = make_registration_surface(
        template_mesh, args.surface_triangles
    )
    print(
        f"  template surface: {template_surface.GetNumberOfPoints():,} points, "
        f"{template_surface.GetNumberOfCells():,} triangles"
    )
    # The original template file is the identity-aligned output.
    shutil.copy2(template_path, args.output_dir / template_path.name)
    del template_mesh

    report: dict[str, Any] = {
        "template": args.template,
        "mapping": "each source heart -> fixed template",
        "transform_type": "rigid rotation and translation only; no scaling",
        "distance_units": "same units as the VTK coordinates",
        "settings": {
            "surface_triangles": args.surface_triangles,
            "maximum_landmarks": args.landmarks,
            "maximum_iterations": args.iterations,
            "convergence_tolerance": args.tolerance,
        },
        "cases": {
            args.template: {
                "matrix": identity_matrix(),
                "iterations": 0,
                "note": "fixed template copied without modification",
            }
        },
    }

    for index, source_path in enumerate(vtk_files, start=1):
        if source_path == template_path:
            continue

        print(f"[{index}/{len(vtk_files)}] Aligning {source_path.name} -> {args.template}")
        source_mesh = read_unstructured_grid(source_path)
        source_surface = make_registration_surface(
            source_mesh, args.surface_triangles
        )
        print(
            f"  source surface: {source_surface.GetNumberOfPoints():,} points, "
            f"{source_surface.GetNumberOfCells():,} triangles"
        )

        before = symmetric_nearest_neighbor_summary(source_surface, template_surface)
        icp = rigid_icp(
            source_surface=source_surface,
            template_surface=template_surface,
            maximum_iterations=args.iterations,
            maximum_landmarks=args.landmarks,
            convergence_tolerance=args.tolerance,
        )
        aligned_surface = transform_polydata(source_surface, icp)
        after = symmetric_nearest_neighbor_summary(aligned_surface, template_surface)

        aligned_mesh = transform_full_mesh(source_mesh, icp)
        output_path = args.output_dir / source_path.name
        write_unstructured_grid(aligned_mesh, output_path)

        iterations = int(icp.GetNumberOfIterations())
        report["cases"][source_path.name] = {
            "matrix": matrix_as_rows(icp),
            "iterations": iterations,
            "icp_transform_change_rms": float(icp.GetMeanDistance()),
            "symmetric_nearest_neighbor_before": before,
            "symmetric_nearest_neighbor_after": after,
            "input_points": int(source_mesh.GetNumberOfPoints()),
            "input_cells": int(source_mesh.GetNumberOfCells()),
            "output": str(output_path),
        }
        print(
            f"  iterations: {iterations}; symmetric NN mean: "
            f"{before['mean']:.4f} -> {after['mean']:.4f}; wrote {output_path}"
        )

        # Release each large volumetric mesh before loading the next heart.
        del source_mesh, source_surface, aligned_surface, aligned_mesh, icp

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved transform and QA report: {report_path}")
    print("Rigid ICP complete. Inspect template/subject overlays before LDDMM.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
