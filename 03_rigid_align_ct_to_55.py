#!/usr/bin/env python3
"""Rigid ICP for roughly co-oriented VTU hearts; default reference ct_case_0055.

Samples boundaries uniformly by area. Minimizes symmetric sampled-point squared
distance using proper rigid Kabsch updates and multiple nearby starts. No scale,
reflection, clipping, or deformation. Distances use the input coordinate units.
Requires numpy, scipy, vtk. Output directory must not already exist.
Active vectors/normals/tensors are rotated. Use --vector-array NAME for otherwise
unmarked physical 3-vector arrays (e.g. fibres); other arrays remain unchanged.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def read(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    if reader.GetErrorCode() or not grid.GetNumberOfCells() or not grid.GetNumberOfPoints():
        raise ValueError(f'Empty/unreadable mesh: {path}')
    if not np.isfinite(vtk_to_numpy(grid.GetPoints().GetData())).all():
        raise ValueError(f'Non-finite coordinates: {path}')
    return grid


def sample_surface(grid, count, seed):
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputData(grid)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(surface.GetOutputPort())
    tri.PassLinesOff(); tri.PassVertsOff(); tri.Update()
    poly = tri.GetOutput()
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    cells = poly.GetPolys()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    if not len(offsets) > 1 or not np.all(np.diff(offsets) == 3):
        raise ValueError('No valid triangular boundary')
    faces = vtk_to_numpy(cells.GetConnectivityArray()).reshape(-1, 3)
    t = points[faces]
    area = np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)/2
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError('Invalid boundary area')
    rng = np.random.default_rng(seed)
    t = t[rng.choice(len(t), count, p=area/area.sum())]
    uv = rng.random((count, 2))
    mask = uv.sum(axis=1) > 1
    uv[mask] = 1-uv[mask]
    return t[:, 0] + uv[:, :1]*(t[:, 1]-t[:, 0]) + uv[:, 1:]*(t[:, 2]-t[:, 0])


def distances(a, b):
    d = np.concatenate([cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]])
    return {'mean': float(d.mean()), 'rmse': float(np.sqrt(np.mean(d*d))),
            'p95': float(np.percentile(d, 95))}


def kabsch(a, b):
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ correction @ u.T
    return r, bc-r @ ac


def icp(a, b, initial_r, initial_t, iterations, tol):
    r, t = initial_r.copy(), initial_t.copy()
    bt = cKDTree(b)
    previous = np.inf
    for step in range(iterations):
        moved = a @ r.T + t
        da, ib = bt.query(moved)
        db, ia = cKDTree(moved).query(b)
        objective = float((np.mean(da*da)+np.mean(db*db))/2)
        if abs(previous-objective) <= tol*max(1.0, previous) and np.isfinite(previous):
            return r, t, step, True
        previous = objective
        dr, dt = kabsch(np.vstack([moved, moved[ia]]), np.vstack([b[ib], b]))
        r, t = dr @ r, dr @ t + dt
    return r, t, iterations, False


def align(a, b, iterations, tol, angle):
    rotations = [np.eye(3)]
    for axis in range(3):
        for sign in [-1, 1]:
            vector = np.zeros(3); vector[axis] = np.deg2rad(sign*angle)
            rotations.append(Rotation.from_rotvec(vector).as_matrix())
    before = distances(a, b)
    best = (np.eye(3), np.zeros(3), before, 0, True, 'unchanged')
    for i, initial_r in enumerate(rotations):
        initial_t = b.mean(axis=0)-initial_r @ a.mean(axis=0)
        r, t, steps, converged = icp(a, b, initial_r, initial_t, iterations, tol)
        score = distances(a @ r.T+t, b)
        if score['rmse'] < best[2]['rmse']:
            best = (r, t, score, steps, converged, f'start_{i}')
    return before, best


def array_signature(data):
    result = []
    for i in range(data.GetNumberOfArrays()):
        a = data.GetAbstractArray(i)
        if isinstance(a, vtk.vtkDataArray):
            payload = vtk_to_numpy(a).tobytes()
        else:
            payload = json.dumps([a.GetVariantValue(j).ToString()
                                  for j in range(a.GetNumberOfValues())]).encode()
        result.append((a.GetName(), a.GetDataType(), a.GetNumberOfComponents(),
                       a.GetNumberOfTuples(), hashlib.sha256(payload).hexdigest()))
    return result


def write_transformed(grid, path, r, t, vector_names):
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-10) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError('Transformation is not a proper rigid rotation')
    out = vtk.vtkUnstructuredGrid(); out.DeepCopy(grid)
    points = vtk.vtkPoints()
    xyz = vtk_to_numpy(grid.GetPoints().GetData()).astype(float) @ r.T+t
    points.SetData(numpy_to_vtk(xyz, deep=True)); out.SetPoints(points)
    unmarked = []
    for kind, data in [('point', out.GetPointData()), ('cell', out.GetCellData())]:
        active = {a.GetName() for a in [data.GetVectors(), data.GetNormals()] if a is not None}
        tensor = data.GetTensors()
        tensor_name = tensor.GetName() if tensor is not None else None
        for i in range(data.GetNumberOfArrays()):
            a = data.GetArray(i)
            if a is None:
                continue
            name, components = a.GetName(), a.GetNumberOfComponents()
            if name in active or name in vector_names:
                if components != 3 or not np.issubdtype(vtk_to_numpy(a).dtype, np.floating):
                    raise ValueError(f'{name}: physical vectors must have 3 floating components')
                values = vtk_to_numpy(a); values[:] = values @ r.T; a.Modified()
            elif name == tensor_name and components == 9:
                values = vtk_to_numpy(a)
                values[:] = (r @ values.reshape(-1, 3, 3) @ r.T).reshape(-1, 9)
                a.Modified()
            elif components > 1:
                unmarked.append(f'{kind}:{name}')
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path)); writer.SetInputData(out)
    if writer.Write() != 1 or writer.GetErrorCode():
        raise RuntimeError(f'Write failed: {path}')
    saved = read(path)
    if not np.array_equal(vtk_to_numpy(saved.GetPoints().GetData()), xyz):
        raise ValueError('Coordinate verification failed')
    for method in ['GetConnectivityArray', 'GetOffsetsArray']:
        if not np.array_equal(vtk_to_numpy(getattr(grid.GetCells(), method)()),
                              vtk_to_numpy(getattr(saved.GetCells(), method)())):
            raise ValueError('Connectivity changed')
    if grid.GetNumberOfCells() != saved.GetNumberOfCells() or any(
            grid.GetCellType(i) != saved.GetCellType(i) for i in range(grid.GetNumberOfCells())):
        raise ValueError('Cell types changed')
    for getter in ['GetPointData', 'GetCellData', 'GetFieldData']:
        if array_signature(getattr(out, getter)()) != array_signature(getattr(saved, getter)()):
            raise ValueError('Saved arrays differ from intended arrays')
    return unmarked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('CenteredCT'))
    p.add_argument('--output-dir', type=Path, default=Path('03.1_ICPAlignedCT55'))
    p.add_argument('--template', default='ct_case_0055.vtu')
    p.add_argument('--expected-count', type=int, default=61)
    p.add_argument('--samples', type=int, default=6000)
    p.add_argument('--iterations', type=int, default=150)
    p.add_argument('--tolerance', type=float, default=1e-7)
    p.add_argument('--start-angle', type=float, default=10)
    p.add_argument('--seed', type=int, default=55)
    p.add_argument('--vector-array', action='append', default=[])
    args = p.parse_args()
    files = sorted(x for x in args.input_dir.glob('ct_case_*.vtu') if x.is_file())
    ref = args.input_dir / args.template
    if len(files) != args.expected_count or not files or ref not in files:
        p.error('Check input count and template filename')
    if args.samples < 100 or args.iterations < 1 or args.tolerance <= 0:
        p.error('Invalid sample count, iterations, or tolerance')
    if args.output_dir.exists():
        p.error('Output directory exists. Choose a new directory; nothing overwritten.')
    reference = read(ref)
    target = sample_surface(reference, args.samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='in_progress', template=str(ref.resolve()),
                  method='multi-start symmetric sampled point-to-point rigid ICP',
                  units='input coordinate units', settings=vars(args).copy(), subjects=[])
    report['settings'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['settings'].items()}
    try:
        for i, file in enumerate(files):
            destination = args.output_dir / file.name
            if file == ref:
                shutil.copy2(file, destination)
                report['subjects'].append(dict(subject=file.stem, fixed_template=True,
                                                matrix=np.eye(4).tolist()))
                print(f'[{i+1}/{len(files)}] {file.stem}: template copied unchanged', flush=True)
                continue
            grid = read(file)
            source = sample_surface(grid, args.samples, args.seed+i+1)
            before, (r, t, after, steps, converged, start) = align(
                source, target, args.iterations, args.tolerance, args.start_angle)
            unmarked = write_transformed(grid, destination, r, t, args.vector_array)
            matrix = np.eye(4); matrix[:3, :3] = r; matrix[:3, 3] = t
            report['subjects'].append(dict(subject=file.stem, before=before, after=after,
                matrix=matrix.tolist(), convention='column: x_new = R x_old + t',
                iterations=steps, converged=converged, selected_start=start,
                unmarked_multicomponent_arrays_unchanged=unmarked, verified=True))
            print(f'[{i+1}/{len(files)}] {file.stem}: sampled RMSE '
                  f'{before["rmse"]:.4f} -> {after["rmse"]:.4f}; verified'
                  + ('; iteration limit reached' if not converged else ''), flush=True)
            if unmarked:
                print('  Check unmarked multicomponent arrays:', ', '.join(unmarked), flush=True)
            (args.output_dir/'icp_report.json').write_text(json.dumps(report, indent=2)+'\n')
        report['status'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'; report['error'] = str(exc)
        raise
    finally:
        (args.output_dir/'icp_report.json').write_text(json.dumps(report, indent=2)+'\n')
    with (args.output_dir/'icp_scores.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['subject', 'before_rmse', 'after_rmse', 'converged'])
        for row in report['subjects']:
            if not row.get('fixed_template'):
                writer.writerow([row['subject'], row['before']['rmse'], row['after']['rmse'], row['converged']])
    print('Done. Inspect overlays before LDDMM. Originals untouched:', args.output_dir)


if __name__ == '__main__':
    main()
