#!/usr/bin/env python3
"""Translate VTU volumes to their surface-area centroid; no other geometry changes.

Requires numpy and vtk. Originals are read-only. A new output directory is
required. Coordinates are saved as float64. All attached arrays are copied
unchanged (coordinate-valued metadata, if present, is NOT reinterpreted).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def read(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if reader.GetErrorCode() or not mesh.GetNumberOfPoints() or not mesh.GetNumberOfCells():
        raise ValueError(f"Unreadable or empty mesh: {path}")
    if not np.isfinite(vtk_to_numpy(mesh.GetPoints().GetData())).all():
        raise ValueError(f"Non-finite coordinates: {path}")
    return mesh


def centroid(mesh):
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputData(mesh)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(surface.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    poly = tri.GetOutput()
    if not poly.GetNumberOfPolys():
        raise ValueError("No boundary triangles")
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    faces = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Non-triangular boundary")
    triangles = points[faces[:, 1:]]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    total = areas.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid boundary area")
    return np.sum(triangles.mean(axis=1) * (areas / total)[:, None], axis=0)


def check_arrays(before, after):
    if before.GetNumberOfArrays() != after.GetNumberOfArrays():
        raise ValueError("Array count changed")
    for i in range(before.GetNumberOfArrays()):
        a, b = before.GetAbstractArray(i), after.GetAbstractArray(i)
        if (a.GetName(), a.GetDataType(), a.GetNumberOfComponents(), a.GetNumberOfTuples()) != (
                b.GetName(), b.GetDataType(), b.GetNumberOfComponents(), b.GetNumberOfTuples()):
            raise ValueError("Array metadata changed")
        if isinstance(a, vtk.vtkDataArray):
            same = np.array_equal(vtk_to_numpy(a), vtk_to_numpy(b), equal_nan=True)
        else:
            same = all(a.GetVariantValue(j).ToString() == b.GetVariantValue(j).ToString()
                       for j in range(a.GetNumberOfValues()))
        if not same:
            raise ValueError(f"Array values changed: {a.GetName()}")


def center_file(source, target):
    original = read(source)
    center = centroid(original)
    xyz = vtk_to_numpy(original.GetPoints().GetData()).astype(np.float64)
    shifted = xyz - center
    output = vtk.vtkUnstructuredGrid()
    output.DeepCopy(original)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(shifted, deep=True))
    output.SetPoints(points)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(target))
    writer.SetInputData(output)
    if writer.Write() != 1 or writer.GetErrorCode():
        raise RuntimeError(f"Write failed: {target}")

    saved = read(target)
    actual = vtk_to_numpy(saved.GetPoints().GetData())
    if not np.array_equal(actual, shifted):
        raise ValueError("Saved coordinates differ from requested translation")
    for a, b in [(original.GetCells().GetConnectivityArray(), saved.GetCells().GetConnectivityArray()),
                 (original.GetCells().GetOffsetsArray(), saved.GetCells().GetOffsetsArray()),
                 (original.GetCellTypesArray(), saved.GetCellTypesArray())]:
        if not np.array_equal(vtk_to_numpy(a), vtk_to_numpy(b)):
            raise ValueError("Cell connectivity/type changed")
    for getter in ('GetPointData', 'GetCellData', 'GetFieldData'):
        check_arrays(getattr(original, getter)(), getattr(saved, getter)())
    residual = centroid(saved)
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(xyz))))
    if np.linalg.norm(residual) > tolerance:
        raise ValueError(f"Centering verification failed: {residual}")
    return dict(source=str(source.resolve()), output=str(target.resolve()),
                original_surface_centroid=center.tolist(), translation=(-center).tolist(),
                centered_surface_centroid=residual.tolist(),
                points=original.GetNumberOfPoints(), cells=original.GetNumberOfCells(),
                verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('MixedCohort'))
    parser.add_argument('--output-dir', type=Path, default=Path('02.1_CenteredCT'))
    parser.add_argument('--expected-count', type=int, default=61)
    args = parser.parse_args()
    sources = sorted(args.input_dir.glob('ct_case_*.vtu'))
    sources = [p for p in sources if p.is_file()]
    if not sources or len(sources) != args.expected_count:
        parser.error(f"Expected {args.expected_count} files, found {len(sources)}")
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new directory. Nothing overwritten.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(operation='translation_only', status='in_progress', files=[])
    report_path = args.output_dir / 'centering_report.json'
    try:
        for i, source in enumerate(sources, 1):
            record = center_file(source, args.output_dir / source.name)
            report['files'].append(record)
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(f"[{i}/{len(sources)}] {source.name}: centered and verified", flush=True)
        report['status'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Done. Originals untouched. Centered copies: {args.output_dir}")


if __name__ == '__main__':
    main()
