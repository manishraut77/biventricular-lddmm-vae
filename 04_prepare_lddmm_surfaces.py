#!/usr/bin/env python3
"""Prepare decimated cardiac surfaces for LDDMM registration.


Run this script from the TemplateMethod directory. By default it processes only
16.vtk, so the fixed-template surface can be inspected before processing the cohort.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import vtk


DEFAULT_INPUT_DIR = Path("ICPAlignedVtkTemplate16")
DEFAULT_OUTPUT_DIR = Path("LDDMMSurfacesTemplate16")
DEFAULT_CASE = "16.vtk"
DEFAULT_TARGET_TRIANGLES = 10_000
DEFAULT_CLEAN_TOLERANCE = 1e-6


def natural_key(path: Path) -> list[object]:
    """Sort 2.vtk before 10.vtk."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def read_unstructured_grid(path: Path) -> vtk.vtkUnstructuredGrid:
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()

    mesh = vtk.vtkUnstructuredGrid()
    mesh.DeepCopy(reader.GetOutput())

    if mesh.GetNumberOfPoints() == 0 or mesh.GetNumberOfCells() == 0:
        raise ValueError(f"VTK file is empty or is not an unstructured grid: {path}")

    cell_types = vtk.vtkCellTypes()
    if hasattr(mesh, "GetDistinctCellTypes"):
        mesh.GetDistinctCellTypes(cell_types)
    else:  # Compatibility with older VTK releases.
        mesh.GetCellTypes(cell_types)
    present = {cell_types.GetCellType(i) for i in range(cell_types.GetNumberOfTypes())}
    if vtk.VTK_TETRA not in present:
        raise ValueError(f"No tetrahedral cells found in {path}")

    return mesh


def triangle_only(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(polydata)
    triangles.PassVertsOff()
    triangles.PassLinesOff()
    triangles.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(triangles.GetOutput())
    return result


def clean_polydata(
    polydata: vtk.vtkPolyData,
    tolerance: float,
) -> vtk.vtkPolyData:
    """KCL's vtkCleanPolyData settings, with converted lines discarded."""
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.SetTolerance(tolerance)
    cleaner.ConvertLinesToPointsOn()
    cleaner.ConvertPolysToLinesOn()
    cleaner.ConvertStripsToPolysOn()
    cleaner.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(cleaner.GetOutput())
    # This is the KCL clean_polydata(..., remove_lines=True) behavior.
    result.SetLines(vtk.vtkCellArray())
    result.SetVerts(vtk.vtkCellArray())
    result.Squeeze()
    return result


def extract_registration_surface(
    volume: vtk.vtkUnstructuredGrid,
    target_triangles: int,
    tolerance: float,
) -> tuple[vtk.vtkPolyData, int]:
    # Exact filter class used by the KCL cohort code.
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(volume)
    surface_filter.Update()

    surface = triangle_only(surface_filter.GetOutput())
    surface = clean_polydata(surface, tolerance)
    surface = triangle_only(surface)
    triangles_before = surface.GetNumberOfPolys()

    if triangles_before > target_triangles:
        # VTK defines target reduction as the fraction removed, not retained.
        reduction = 1.0 - (target_triangles / triangles_before)
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputData(surface)
        decimator.VolumePreservationOn()
        decimator.SetTargetReduction(reduction)
        decimator.Update()

        surface = triangle_only(decimator.GetOutput())
        surface = clean_polydata(surface, tolerance)
        surface = triangle_only(surface)

    return surface, triangles_before


def count_feature_edges(polydata: vtk.vtkPolyData, kind: str) -> int:
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(polydata)
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    if kind == "boundary":
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOff()
    elif kind == "non_manifold":
        edges.BoundaryEdgesOff()
        edges.NonManifoldEdgesOn()
    else:
        raise ValueError(f"Unknown edge kind: {kind}")
    edges.Update()
    return edges.GetOutput().GetNumberOfCells()


def count_connected_regions(polydata: vtk.vtkPolyData) -> int:
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOff()
    connectivity.Update()
    return connectivity.GetNumberOfExtractedRegions()


def write_polydata(polydata: vtk.vtkPolyData, path: Path) -> None:
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.SetFileTypeToASCII()
    if writer.Write() != 1:
        raise OSError(f"VTK failed to write {path}")


def process_case(
    input_path: Path,
    output_path: Path,
    target_triangles: int,
    tolerance: float,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    print(f"Preparing {input_path.name}")
    volume = read_unstructured_grid(input_path)
    print(
        f"  volume: {volume.GetNumberOfPoints():,} points, "
        f"{volume.GetNumberOfCells():,} cells"
    )

    surface, triangles_before = extract_registration_surface(
        volume, target_triangles, tolerance
    )
    triangles_after = surface.GetNumberOfPolys()

    if triangles_after == 0:
        raise ValueError(f"Surface extraction produced no triangles for {input_path}")
    if surface.GetNumberOfLines() != 0 or surface.GetNumberOfVerts() != 0:
        raise ValueError(f"Non-triangle cells remain in the output for {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_polydata(surface, output_path)

    print(f"  extracted surface: {triangles_before:,} triangles")
    print(
        f"  registration surface: {surface.GetNumberOfPoints():,} points, "
        f"{triangles_after:,} triangles"
    )
    print(
        "  QA: "
        f"{count_connected_regions(surface)} connected region(s), "
        f"{count_feature_edges(surface, 'boundary')} boundary edge(s), "
        f"{count_feature_edges(surface, 'non_manifold')} non-manifold edge(s)"
    )
    print(f"  wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract, clean, and decimate LDDMM surfaces from rigidly aligned "
            "tetrahedral heart meshes."
        )
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"input directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--case", default=DEFAULT_CASE,
        help=f"single VTK filename to process (default: {DEFAULT_CASE})",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="process every .vtk file in the input directory",
    )
    parser.add_argument(
        "--target-triangles", type=int, default=DEFAULT_TARGET_TRIANGLES,
        help=f"desired triangle count after decimation (default: {DEFAULT_TARGET_TRIANGLES})",
    )
    parser.add_argument(
        "--clean-tolerance", type=float, default=DEFAULT_CLEAN_TOLERANCE,
        help=f"relative vtkCleanPolyData tolerance (default: {DEFAULT_CLEAN_TOLERANCE})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing output surface files",
    )
    args = parser.parse_args()

    if args.target_triangles < 100:
        parser.error("--target-triangles must be at least 100")
    if args.clean_tolerance < 0:
        parser.error("--clean-tolerance cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    if args.all:
        input_paths = sorted(args.input_dir.glob("*.vtk"), key=natural_key)
    else:
        input_paths = [args.input_dir / args.case]

    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        print(f"ERROR: input file not found: {missing[0]}", file=sys.stderr)
        return 1
    if not input_paths:
        print(f"ERROR: no .vtk files found in {args.input_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(input_paths)} case(s)")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")

    try:
        for index, input_path in enumerate(input_paths, start=1):
            output_path = args.output_dir / f"{input_path.stem}_surface.vtk"
            print(f"[{index}/{len(input_paths)}]")
            process_case(
                input_path=input_path,
                output_path=output_path,
                target_triangles=args.target_triangles,
                tolerance=args.clean_tolerance,
                overwrite=args.overwrite,
            )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Surface preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
