#!/usr/bin/env python3
"""Extract full exterior triangular surfaces from aligned CT VTU volumes.

Inputs are read-only. Every output is legacy VTK PolyData with triangles only.
The output directory must not already exist. Target surfaces may have different
point and triangle counts; later registrations inherit the template topology.

Requires: numpy, vtk
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_volume(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    if reader.GetErrorCode() or not grid.GetNumberOfPoints() or not grid.GetNumberOfCells():
        raise ValueError(f"Unreadable or empty VTU: {path}")
    coordinates = vtk_to_numpy(grid.GetPoints().GetData())
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Non-finite coordinates: {path}")
    return grid


def extract_surface(grid):
    boundary = vtk.vtkDataSetSurfaceFilter()
    boundary.SetInputData(grid)
    boundary.PassThroughPointIdsOn()
    boundary.SetOriginalPointIdsName("VolumePointIndex")
    boundary.PassThroughCellIdsOn()
    boundary.SetOriginalCellIdsName("VolumeCellIndex")

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(boundary.GetOutputPort())
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()

    surface = vtk.vtkPolyData()
    surface.DeepCopy(triangles.GetOutput())
    if not surface.GetNumberOfPoints() or not surface.GetNumberOfPolys():
        raise ValueError("Surface extraction returned no triangular surface")
    return surface


def arrays(surface):
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(np.float64)
    cells = surface.GetPolys()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    if len(offsets) != surface.GetNumberOfPolys() + 1 or not np.all(np.diff(offsets) == 3):
        raise ValueError("Extracted surface contains non-triangular polygons")
    faces = vtk_to_numpy(cells.GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("Surface connectivity references an invalid point")
    return points, faces


def edge_counts(faces):
    edges = np.sort(
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    _, multiplicity = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(multiplicity == 1)), int(np.count_nonzero(multiplicity > 2))


def metrics(surface):
    points, faces = arrays(surface)
    vertices = points[faces]
    twice_area = np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=1,
    )
    if not np.isfinite(twice_area).all() or np.any(twice_area <= 0):
        raise ValueError("Surface contains a degenerate or invalid triangle")
    boundary_edges, nonmanifold_edges = edge_counts(faces)

    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(surface)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOff()
    connectivity.Update()

    return {
        "points": int(len(points)),
        "triangles": int(len(faces)),
        "surface_area": float(twice_area.sum() / 2),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "connected_components": int(connectivity.GetNumberOfExtractedRegions()),
        "bounds": [float(value) for value in surface.GetBounds()],
    }


def write_surface(path, surface):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"Could not write {path}")

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    saved = reader.GetOutput()
    before_points, before_faces = arrays(surface)
    after_points, after_faces = arrays(saved)
    if not np.array_equal(before_points, after_points):
        raise ValueError(f"Saved coordinates differ from extracted coordinates: {path}")
    if not np.array_equal(before_faces, after_faces):
        raise ValueError(f"Saved triangle connectivity differs: {path}")
    if saved.GetNumberOfLines() or saved.GetNumberOfVerts() or saved.GetNumberOfStrips():
        raise ValueError(f"Unexpected non-triangle cells in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("ICPAlignedCT55"))
    parser.add_argument("--output-dir", type=Path, default=Path("04.1_CTRegistrationSurfaces"))
    parser.add_argument("--expected-count", type=int, default=61)
    args = parser.parse_args()

    files = sorted(path for path in args.input_dir.glob("ct_case_*.vtu") if path.is_file())
    if len(files) != args.expected_count:
        parser.error(f"Expected {args.expected_count} VTU files, found {len(files)}")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for index, source in enumerate(files, start=1):
            grid = read_volume(source)
            extracted = extract_surface(grid)
            result = metrics(extracted)
            destination = args.output_dir / f"{source.stem}.vtk"
            write_surface(destination, extracted)
            record = {
                "subject": source.stem,
                "source": str(source.resolve()),
                "surface": str(destination.resolve()),
                "source_sha256": sha256(source),
                **result,
            }
            records.append(record)
            print(
                f"[{index:02d}/{len(files):02d}] {source.stem}: "
                f"{result['points']:,} points, {result['triangles']:,} triangles; verified",
                flush=True,
            )

        fieldnames = list(records[0])
        with (args.output_dir / "surface_manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        (args.output_dir / "surface_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "representation": "legacy VTK PolyData; triangle cells only",
                    "count": len(records),
                    "same_triangle_count_required_for_registration": False,
                    "subjects": records,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    except Exception:
        (args.output_dir / "INCOMPLETE.txt").write_text(
            "Extraction stopped before all requested surfaces passed verification.\n"
        )
        raise

    print(f"Done. Triangular surfaces: {args.output_dir}")


if __name__ == "__main__":
    main()
