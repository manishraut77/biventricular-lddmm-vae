#!/usr/bin/env python3
"""Prepare and register the complete CT cohort to CT55 in one script.

Compared with the original pilot, this version:

* removes the artificial planar cap from the anatomical attachment term;
* splits the remaining surface into connected anatomical objects;
* matches every object and its basal contour under one shared diffeomorphism;
* first reaches the endpoint plane constraint, then refines anatomy at a fixed
  penalty instead of stopping immediately when the base becomes flat;
* selects the lowest anatomical objective among feasible iterates; and
* measures symmetric surface error in physical input units before PASS.

The ``all`` command runs the same multi-object, flat-base constrained LDDMM
algorithm for every subject. It publishes flat registered surfaces directly in
``CTRegistered`` and the corresponding compressed momentum archives in
``Moments``. Preparation data and per-case QA reports are retained beneath
``CTRegistered/RegistrationMetadata`` so the run is reproducible and resumable.

The lower-level ``prepare``, ``register`` and ``replay`` commands are retained
for diagnosis and single-case work. The saved custom momenta are replayable by
this script, but are not Deformetrica-compatible. Arbitrary VAE-sampled
momenta still require endpoint refinement or deterministic planar recapping.

Requires: numpy, vtk, torch
"""

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_surface(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    surface = reader.GetOutput()
    if not surface.GetNumberOfPoints() or not surface.GetNumberOfPolys():
        raise ValueError(f"Unreadable or empty surface: {path}")
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(np.float64)
    cells = surface.GetPolys()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    if len(offsets) != surface.GetNumberOfPolys() + 1 or not np.all(np.diff(offsets) == 3):
        raise ValueError(f"Surface is not triangle-only: {path}")
    faces = vtk_to_numpy(cells.GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    if not np.isfinite(points).all() or faces.min() < 0 or faces.max() >= len(points):
        raise ValueError(f"Invalid points or connectivity: {path}")
    return points, faces


def make_polydata(points, faces):
    surface = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    surface.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.asarray(faces, dtype=np.int64).ravel(), deep=True),
    )
    surface.SetPolys(cells)
    return surface


def write_surface(path, points, faces, basal_faces=None, object_labels=None):
    surface = make_polydata(points, faces)
    if basal_faces is not None:
        labels = numpy_to_vtk(np.asarray(basal_faces, dtype=np.uint8), deep=True)
        labels.SetName("BasalFace")
        surface.GetCellData().AddArray(labels)
    if object_labels is not None:
        labels = numpy_to_vtk(np.asarray(object_labels, dtype=np.int32), deep=True)
        labels.SetName("AnatomicalObject")
        surface.GetCellData().AddArray(labels)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"Could not write {path}")
    saved_points, saved_faces = read_surface(path)
    if not np.array_equal(saved_points, points) or not np.array_equal(saved_faces, faces):
        raise ValueError(f"Saved surface differs from intended surface: {path}")


def basal_patch(points, faces, plane_tolerance, max_tilt):
    triangles = points[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(twice_area[:, None], 1e-30)
    candidates = np.flatnonzero(
        (np.abs(normals[:, 2]) >= np.cos(np.deg2rad(max_tilt))) & (twice_area > 0)
    )
    if len(candidates) < 10:
        raise ValueError("No plausible basal plane; check mesh orientation")

    rng = np.random.default_rng(55)
    trials = rng.choice(
        candidates,
        min(300, len(candidates)),
        replace=False,
        p=twice_area[candidates] / twice_area[candidates].sum(),
    )
    best_mask = None
    best_area = -1.0
    for triangle_index in trials:
        normal = normals[triangle_index]
        offset = float(normal @ triangles[triangle_index, 0])
        residual = np.max(np.abs(np.einsum("fvc,c->fv", triangles, normal) - offset), axis=1)
        mask = residual <= plane_tolerance
        mask &= np.abs(normals @ normal) >= np.cos(np.deg2rad(8.0))
        area = float(twice_area[mask].sum())
        if area > best_area:
            best_area, best_mask = area, mask

    for _ in range(3):
        basal_ids = np.unique(faces[best_mask])
        cloud = points[basal_ids]
        center = cloud.mean(axis=0)
        _, _, vh = np.linalg.svd(cloud - center, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        offset = float(normal @ center)
        residual = np.max(np.abs(np.einsum("fvc,c->fv", triangles, normal) - offset), axis=1)
        best_mask = residual <= plane_tolerance
        best_mask &= np.abs(normals @ normal) >= np.cos(np.deg2rad(8.0))

    basal_ids = np.unique(faces[best_mask])
    errors = np.einsum("ij,j->i", points[basal_ids], normal) - offset
    area_fraction = float(twice_area[best_mask].sum() / twice_area.sum())
    if len(basal_ids) < 20 or not 0.005 < area_fraction < 0.4:
        raise ValueError(
            f"Unreliable basal patch: {len(basal_ids)} vertices, area fraction {area_fraction}"
        )
    qa = {
        "basal_vertices": int(len(basal_ids)),
        "basal_triangles": int(best_mask.sum()),
        "basal_area_fraction": area_fraction,
        "original_max_plane_error": float(np.max(np.abs(errors))),
        "original_rms_plane_error": float(np.sqrt(np.mean(errors**2))),
        "plane_normal": normal.tolist(),
        "plane_offset": offset,
    }
    return best_mask, basal_ids, normal, offset, qa


def triangle_areas(points, faces):
    triangles = points[faces]
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )


def face_components(faces):
    """Return triangle components joined through complete mesh edges."""
    parent = np.arange(len(faces), dtype=np.int64)
    rank = np.zeros(len(faces), dtype=np.uint8)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1

    owner = {}
    for face_index, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            previous = owner.get(edge)
            if previous is None:
                owner[edge] = face_index
            else:
                union(face_index, previous)

    groups = {}
    for face_index in range(len(faces)):
        groups.setdefault(find(face_index), []).append(face_index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def label_anatomical_objects(points, faces, basal_faces, expected_objects):
    """Label non-basal connected components by descending physical area."""
    anatomical_face_ids = np.flatnonzero(~np.asarray(basal_faces, dtype=bool))
    components = face_components(faces[anatomical_face_ids])
    areas = triangle_areas(points, faces)
    components = sorted(
        (anatomical_face_ids[component] for component in components),
        key=lambda indices: float(areas[indices].sum()),
        reverse=True,
    )
    if len(components) != expected_objects:
        counts = [int(len(component)) for component in components]
        raise ValueError(
            f"Expected {expected_objects} non-basal anatomical objects after removing "
            f"the cap, found {len(components)} with triangle counts {counts}. "
            "Review the BasalFace label before registration."
        )

    labels = np.zeros(len(faces), dtype=np.int32)
    summaries = []
    for label, face_ids in enumerate(components, start=1):
        labels[face_ids] = label
        vertex_ids = np.unique(faces[face_ids])
        summaries.append(
            {
                "label": label,
                "triangles": int(len(face_ids)),
                "vertices": int(len(vertex_ids)),
                "area": float(areas[face_ids].sum()),
                "centroid": points[vertex_ids].mean(axis=0).tolist(),
            }
        )
    if np.any((~basal_faces) & (labels == 0)):
        raise RuntimeError("Some non-basal triangles were not assigned to an object")
    return labels, summaries


def compact_submesh(points, faces):
    vertex_ids, inverse = np.unique(faces, return_inverse=True)
    return points[vertex_ids], inverse.reshape(-1, 3), vertex_ids


def boundary_edges(faces):
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    ordered = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(ordered, axis=0, return_counts=True)
    result = unique_edges[counts == 1]
    if not len(result):
        raise ValueError("An anatomical object has no basal boundary edges")
    return result.astype(np.int64)


def prepare(args):
    files = sorted(path for path in args.surface_dir.glob("ct_case_*.vtk") if path.is_file())
    if len(files) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} surfaces, found {len(files)}")
    if not any(path.stem == args.template for path in files):
        raise ValueError(f"Template {args.template}.vtk is missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for index, source in enumerate(files, start=1):
            points, faces = read_surface(source)
            mask, basal_ids, normal, offset, qa = basal_patch(
                points, faces, args.plane_tolerance, args.max_tilt
            )
            object_labels, object_summaries = label_anatomical_objects(
                points, faces, mask, args.expected_objects
            )
            np.savez_compressed(
                args.output_dir / f"{source.stem}.npz",
                points=points,
                faces=faces,
                basal_faces=mask,
                basal_ids=basal_ids,
                object_labels=object_labels,
                plane_normal=normal,
                plane_offset=offset,
            )
            write_surface(
                args.output_dir / f"{source.stem}_review_labels.vtk",
                points,
                faces,
                mask,
                object_labels,
            )
            records.append(
                {
                    "subject": source.stem,
                    "source_surface": str(source.resolve()),
                    "source_sha256": sha256(source),
                    "points": int(len(points)),
                    "triangles": int(len(faces)),
                    "anatomical_objects": object_summaries,
                    **qa,
                }
            )
            print(
                f"[{index:02d}/{len(files):02d}] {source.stem}: "
                f"{qa['basal_vertices']} basal vertices; "
                f"{len(object_summaries)} anatomical objects; review labels written",
                flush=True,
            )
        save_json(
            args.output_dir / "manifest.json",
            {
                "status": "labels_require_visual_review",
                "template": args.template,
                "units": "input coordinate units",
                "object_label_meaning": {
                    "0": "artificial basal cap",
                    "1..N": "non-basal components sorted by descending surface area",
                },
                "subjects": records,
            },
        )
    except Exception:
        (args.output_dir / "INCOMPLETE.txt").write_text("Preparation did not finish.\n")
        raise


def torch_setup():
    import torch

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    return torch


def kernel(torch, x, y, width):
    return torch.exp(-((x[:, None, :] - y[None, :, :]) ** 2).sum(-1) / width**2)


def hamiltonian_rhs(torch, control, momentum, shape, width):
    delta = control[:, None, :] - control[None, :, :]
    matrix = torch.exp(-(delta**2).sum(-1) / width**2)
    d_control = matrix @ momentum
    d_momentum = (
        (2.0 / width**2)
        * ((matrix * (momentum @ momentum.T))[:, :, None] * delta).sum(1)
    )
    d_shape = kernel(torch, shape, control, width) @ momentum
    return d_control, d_momentum, d_shape


def shoot(torch, control, momentum, shape, width, steps):
    dt = 1.0 / steps
    for _ in range(steps):
        dc, dm, ds = hamiltonian_rhs(torch, control, momentum, shape, width)
        dc2, dm2, ds2 = hamiltonian_rhs(
            torch,
            control + dt * dc / 2,
            momentum + dt * dm / 2,
            shape + dt * ds / 2,
            width,
        )
        control = control + dt * dc2
        momentum = momentum + dt * dm2
        shape = shape + dt * ds2
    return shape


def shoot_chunks(torch, control, momentum, points, width, steps):
    with torch.no_grad():
        pieces = [
            shoot(torch, control, momentum, torch.tensor(points[i : i + 2048]), width, steps)
            .cpu()
            .numpy()
            for i in range(0, len(points), 2048)
        ]
    return np.concatenate(pieces)


def decimate(points, faces, target_triangles):
    surface = make_polydata(points, faces)
    if len(faces) > target_triangles:
        reduction = vtk.vtkDecimatePro()
        reduction.SetInputData(surface)
        reduction.SetTargetReduction(1.0 - target_triangles / len(faces))
        reduction.PreserveTopologyOn()
        reduction.SplittingOff()
        reduction.BoundaryVertexDeletionOff()
        reduction.Update()
        surface = reduction.GetOutput()
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(np.float64)
    faces = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    return points, faces


def anatomical_components(dataset):
    if "object_labels" not in dataset.files:
        raise ValueError(
            "Prepared data has no AnatomicalObject labels. Rerun this script's "
            "prepare command into a new directory and visually review the labels."
        )
    points = dataset["points"]
    faces = dataset["faces"]
    labels = dataset["object_labels"]
    components = []
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        local_points, local_faces, original_ids = compact_submesh(points, faces[labels == label])
        edges = boundary_edges(local_faces)
        areas = triangle_areas(local_points, local_faces)
        components.append(
            {
                "label": label,
                "points": local_points,
                "faces": local_faces,
                "boundary_edges": edges,
                "original_vertex_ids": original_ids,
                "area": float(areas.sum()),
                "centroid": np.average(
                    local_points[local_faces].mean(axis=1), weights=np.maximum(areas, 1e-30), axis=0
                ),
            }
        )
    return components


def pair_components(template_components, target_components, diagonal):
    if len(template_components) != len(target_components):
        raise ValueError(
            f"Object-count mismatch: template has {len(template_components)}, "
            f"target has {len(target_components)}"
        )
    best = None
    for permutation in itertools.permutations(range(len(target_components))):
        score = 0.0
        for template_component, target_index in zip(template_components, permutation):
            target_component = target_components[target_index]
            score += float(
                np.linalg.norm(template_component["centroid"] - target_component["centroid"])
                / max(diagonal, 1e-12)
                + 0.25
                * abs(np.log(template_component["area"] / target_component["area"]))
            )
        if best is None or score < best[0]:
            best = (score, permutation)
    return [
        (template_component, target_components[target_index])
        for template_component, target_index in zip(template_components, best[1])
    ], float(best[0])


def triangle_budgets(component_pairs, total):
    weights = np.asarray(
        [len(template["faces"]) + len(target["faces"]) for template, target in component_pairs],
        dtype=float,
    )
    raw = total * weights / weights.sum()
    budgets = np.maximum(100, np.rint(raw).astype(int))
    return budgets.tolist()


def varifold_features(torch, points, faces):
    triangles = points[faces]
    cross = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    magnitude = torch.sqrt((cross**2).sum(1) + 1e-24)
    return triangles.mean(1), cross / magnitude[:, None], magnitude / 2


def varifold_inner(torch, left, right, width):
    center_a, normal_a, area_a = left
    center_b, normal_b, area_b = right
    return (
        kernel(torch, center_a, center_b, width)
        * (normal_a @ normal_b.T) ** 2
        * area_a[:, None]
        * area_b[None, :]
    ).sum()


def curve_varifold_features(torch, points, edges):
    segments = points[edges]
    vectors = segments[:, 1] - segments[:, 0]
    lengths = torch.sqrt((vectors**2).sum(1) + 1e-24)
    return segments.mean(1), vectors / lengths[:, None], lengths


def curve_varifold_inner(torch, left, right, width):
    center_a, tangent_a, length_a = left
    center_b, tangent_b, length_b = right
    return (
        kernel(torch, center_a, center_b, width)
        * (tangent_a @ tangent_b.T) ** 2
        * length_a[:, None]
        * length_b[None, :]
    ).sum()


def normalized_varifold_distance(
    torch, moved, target, width, inner_function, target_self=None
):
    if target_self is None:
        target_self = inner_function(torch, target, target, width)
    return (
        inner_function(torch, moved, moved, width)
        + target_self
        - 2.0 * inner_function(torch, moved, target, width)
    ) / torch.clamp(target_self, min=1e-24)


def point_to_surface_distances(query_points, surface_points, surface_faces):
    distance = vtk.vtkImplicitPolyDataDistance()
    distance.SetInput(make_polydata(surface_points, surface_faces))
    values = np.empty(len(query_points), dtype=np.float64)
    for index, point in enumerate(query_points):
        values[index] = abs(distance.EvaluateFunction(point))
    return values


def symmetric_surface_metrics(left_points, left_faces, right_points, right_faces):
    distances = np.concatenate(
        (
            point_to_surface_distances(left_points, right_points, right_faces),
            point_to_surface_distances(right_points, left_points, left_faces),
        )
    )
    if not np.isfinite(distances).all():
        raise FloatingPointError("Non-finite surface distance")
    return {
        "mean": float(distances.mean()),
        "rms": float(np.sqrt(np.mean(distances**2))),
        "p95": float(np.percentile(distances, 95)),
        "maximum": float(distances.max()),
    }


def surface_qa(original_points, deformed_points, faces):
    before = original_points[faces]
    after = deformed_points[faces]
    area_before = np.linalg.norm(
        np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0]), axis=1
    )
    area_after = np.linalg.norm(
        np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0]), axis=1
    )
    ratio = area_after / np.maximum(area_before, 1e-30)
    return {
        "degenerate_triangles": int(np.count_nonzero(area_after <= 1e-14)),
        "minimum_triangle_area_ratio": float(ratio.min()),
        "maximum_triangle_area_ratio": float(ratio.max()),
    }


def register(args):
    torch = torch_setup()
    if not args.reviewed_base_labels:
        raise ValueError(
            "Review every BasalFace and AnatomicalObject label, then pass "
            "--reviewed-base-labels"
        )
    manifest = json.loads((args.prepared_dir / "manifest.json").read_text())
    subjects = {record["subject"]: record for record in manifest["subjects"]}
    template_name = manifest["template"]
    if args.target not in subjects:
        raise ValueError(f"Unknown target: {args.target}")
    for name in (template_name, args.target):
        source = Path(subjects[name]["source_surface"])
        if sha256(source) != subjects[name]["source_sha256"]:
            raise ValueError(f"Prepared source changed: {source}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    template = np.load(args.prepared_dir / f"{template_name}.npz")
    target = np.load(args.prepared_dir / f"{args.target}.npz")
    template_components = anatomical_components(template)
    target_components = anatomical_components(target)
    diagonal = float(np.linalg.norm(np.ptp(template["points"], axis=0)))
    component_pairs, pairing_score = pair_components(
        template_components, target_components, diagonal
    )
    budgets = triangle_budgets(component_pairs, args.fitting_triangles)

    origin = template["points"].mean(0)
    scale = args.kernel_width
    axes = [
        np.arange(low, high + args.control_spacing * 0.5, args.control_spacing)
        for low, high in zip(template["points"].min(0), template["points"].max(0))
    ]
    control_points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    control = torch.tensor((control_points - origin) / scale)
    momentum = torch.zeros_like(control, requires_grad=True)

    shape_parts = []
    surface_models = []
    curve_models = []
    pair_report = []
    point_cursor = 0
    total_template_fit_triangles = 0
    total_target_fit_triangles = 0
    for pair_index, ((template_component, target_component), budget) in enumerate(
        zip(component_pairs, budgets), start=1
    ):
        template_fit_points, template_fit_faces = decimate(
            template_component["points"], template_component["faces"], budget
        )
        target_fit_points, target_fit_faces = decimate(
            target_component["points"], target_component["faces"], budget
        )
        start = point_cursor
        stop = start + len(template_fit_points)
        shape_parts.append(template_fit_points)
        point_cursor = stop
        target_fit_points_tensor = torch.tensor((target_fit_points - origin) / scale)
        target_fit_faces_tensor = torch.tensor(target_fit_faces)
        surface_models.append(
            {
                "name": f"object_{pair_index}",
                "slice": slice(start, stop),
                "template_faces": torch.tensor(template_fit_faces),
                "target_features": varifold_features(
                    torch, target_fit_points_tensor, target_fit_faces_tensor
                ),
            }
        )
        total_template_fit_triangles += len(template_fit_faces)
        total_target_fit_triangles += len(target_fit_faces)

        template_boundary_ids, template_boundary_inverse = np.unique(
            template_component["boundary_edges"], return_inverse=True
        )
        target_boundary_ids, target_boundary_inverse = np.unique(
            target_component["boundary_edges"], return_inverse=True
        )
        template_boundary_points = template_component["points"][template_boundary_ids]
        target_boundary_points = target_component["points"][target_boundary_ids]
        template_boundary_edges = template_boundary_inverse.reshape(-1, 2)
        target_boundary_edges = target_boundary_inverse.reshape(-1, 2)
        curve_start = point_cursor
        curve_stop = curve_start + len(template_boundary_points)
        shape_parts.append(template_boundary_points)
        point_cursor = curve_stop
        curve_models.append(
            {
                "name": f"object_{pair_index}_basal_contour",
                "slice": slice(curve_start, curve_stop),
                "template_edges": torch.tensor(template_boundary_edges),
                "target_features": curve_varifold_features(
                    torch,
                    torch.tensor((target_boundary_points - origin) / scale),
                    torch.tensor(target_boundary_edges),
                ),
            }
        )
        pair_report.append(
            {
                "template_label": int(template_component["label"]),
                "target_label": int(target_component["label"]),
                "template_triangles": int(len(template_component["faces"])),
                "target_triangles": int(len(target_component["faces"])),
                "template_fitting_triangles": int(len(template_fit_faces)),
                "target_fitting_triangles": int(len(target_fit_faces)),
                "template_boundary_segments": int(len(template_boundary_edges)),
                "target_boundary_segments": int(len(target_boundary_edges)),
            }
        )

    base_points = template["points"][template["basal_ids"]]
    base_start = point_cursor
    base_stop = base_start + len(base_points)
    shape_parts.append(base_points)
    shape = torch.tensor((np.vstack(shape_parts) - origin) / scale)
    plane_normal = torch.tensor(target["plane_normal"])
    plane_offset = float(
        (target["plane_offset"] - np.dot(origin, target["plane_normal"])) / scale
    )

    def stage_context(attachment_width):
        surface_width = attachment_width / scale
        curve_width = args.contour_width / scale
        return {
            "surface_width": surface_width,
            "curve_width": curve_width,
            "surface_target_self": [
                varifold_inner(
                    torch,
                    model["target_features"],
                    model["target_features"],
                    surface_width,
                ).detach()
                for model in surface_models
            ],
            "curve_target_self": [
                curve_varifold_inner(
                    torch,
                    model["target_features"],
                    model["target_features"],
                    curve_width,
                ).detach()
                for model in curve_models
            ],
        }

    def evaluate(momentum_value, context):
        moved = shoot(torch, control, momentum_value, shape, 1.0, args.steps)
        surface_terms = []
        for model, target_self in zip(
            surface_models, context["surface_target_self"]
        ):
            moved_features = varifold_features(
                torch, moved[model["slice"]], model["template_faces"]
            )
            surface_terms.append(
                normalized_varifold_distance(
                    torch,
                    moved_features,
                    model["target_features"],
                    context["surface_width"],
                    varifold_inner,
                    target_self,
                )
            )
        curve_terms = []
        for model, target_self in zip(curve_models, context["curve_target_self"]):
            moved_features = curve_varifold_features(
                torch, moved[model["slice"]], model["template_edges"]
            )
            curve_terms.append(
                normalized_varifold_distance(
                    torch,
                    moved_features,
                    model["target_features"],
                    context["curve_width"],
                    curve_varifold_inner,
                    target_self,
                )
            )
        surface_attachment = torch.stack(surface_terms).mean()
        contour_attachment = torch.stack(curve_terms).mean()
        attachment = surface_attachment + args.contour_weight * contour_attachment
        energy = 0.5 * (
            momentum_value
            * (kernel(torch, control, control, 1.0) @ momentum_value)
        ).sum()
        constraint = moved[base_start:base_stop] @ plane_normal - plane_offset
        return {
            "moved": moved,
            "attachment": attachment,
            "surface_attachment": surface_attachment,
            "contour_attachment": contour_attachment,
            "energy": energy,
            "constraint": constraint,
        }

    contexts = {
        float(width): stage_context(float(width)) for width in args.attachment_widths
    }
    with torch.no_grad():
        initial_values = evaluate(momentum, contexts[float(args.attachment_widths[0])])
        initial_attachment = float(initial_values["attachment"])
    multiplier = torch.zeros(len(base_points))
    penalty = args.initial_penalty
    optimization_flat_tolerance = args.flat_tolerance * args.optimization_flat_fraction
    history = []
    best_infeasible = None
    best_feasible = None
    best_final_feasible = None
    final_width = float(args.attachment_widths[-1])

    def consider_candidate(values, phase, width):
        nonlocal best_infeasible, best_feasible, best_final_feasible
        plane_error = float(values["constraint"].abs().max() * scale)
        anatomical_objective = float(
            values["attachment"] + args.energy_weight * values["energy"]
        )
        candidate = {
            "momentum": momentum.detach().clone(),
            "maximum_plane_error": plane_error,
            "anatomical_objective": anatomical_objective,
            "normalized_attachment": float(values["attachment"]),
            "surface_attachment": float(values["surface_attachment"]),
            "contour_attachment": float(values["contour_attachment"]),
            "energy": float(values["energy"]),
            "phase": phase,
            "attachment_width": float(width),
        }
        if best_infeasible is None or plane_error < best_infeasible["maximum_plane_error"]:
            best_infeasible = candidate
        if plane_error <= optimization_flat_tolerance:
            if (
                best_feasible is None
                or anatomical_objective < best_feasible["anatomical_objective"]
            ):
                best_feasible = candidate
            if float(width) == final_width and (
                best_final_feasible is None
                or anatomical_objective < best_final_feasible["anatomical_objective"]
            ):
                best_final_feasible = candidate

    def optimize_block(phase, block_index, width, iterations):
        context = contexts[float(width)]
        optimizer = torch.optim.LBFGS(
            [momentum],
            lr=args.learning_rate,
            max_iter=iterations,
            line_search_fn="strong_wolfe",
            tolerance_grad=args.gradient_tolerance,
            tolerance_change=args.parameter_tolerance,
        )

        def closure():
            optimizer.zero_grad()
            values = evaluate(momentum, context)
            constraint = values["constraint"]
            loss = (
                values["attachment"]
                + args.energy_weight * values["energy"]
                + (multiplier * constraint).mean()
                + 0.5 * penalty * (constraint**2).mean()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite optimization objective")
            loss.backward()
            return loss

        optimizer.step(closure)
        gradient_norm = None
        if momentum.grad is not None and torch.isfinite(momentum.grad).all():
            gradient_norm = float(momentum.grad.norm())
        with torch.no_grad():
            values = evaluate(momentum, context)
            maximum_error = float(values["constraint"].abs().max() * scale)
            row = {
                "phase": phase,
                "block": int(block_index),
                "attachment_width": float(width),
                "normalized_attachment": float(values["attachment"]),
                "surface_attachment": float(values["surface_attachment"]),
                "contour_attachment": float(values["contour_attachment"]),
                "energy": float(values["energy"]),
                "maximum_plane_error": maximum_error,
                "penalty": float(penalty),
                "gradient_norm": gradient_norm,
            }
            history.append(row)
            print(row, flush=True)
            consider_candidate(values, phase, width)
            return values, row

    print(
        f"{len(control_points)} shared control points; "
        f"{total_template_fit_triangles}/{total_target_fit_triangles} fitting triangles; "
        f"{len(component_pairs)} paired anatomical objects; "
        f"{len(base_points)} constrained basal vertices",
        flush=True,
    )
    coarse_width = float(args.attachment_widths[0])
    for outer in range(args.outer_iterations):
        values, row = optimize_block(
            "constraint", outer + 1, coarse_width, args.inner_iterations
        )
        with torch.no_grad():
            multiplier += penalty * values["constraint"]
            if row["maximum_plane_error"] <= optimization_flat_tolerance:
                break
            penalty = min(penalty * args.penalty_growth, args.maximum_penalty)

    # Continue anatomical optimization after feasibility. The penalty no longer
    # increases; multiplier updates keep the basal vertices close to the plane.
    for width in (float(value) for value in args.attachment_widths):
        previous_attachment = None
        stable_rounds = 0
        for refinement_round in range(1, args.refinement_rounds + 1):
            values, row = optimize_block(
                "anatomy_refinement",
                refinement_round,
                width,
                args.refinement_iterations,
            )
            with torch.no_grad():
                multiplier += penalty * values["constraint"]
            current_attachment = row["normalized_attachment"]
            if previous_attachment is not None:
                relative_change = abs(previous_attachment - current_attachment) / max(
                    abs(previous_attachment), 1e-12
                )
                row["relative_attachment_change"] = float(relative_change)
                if (
                    row["maximum_plane_error"] <= optimization_flat_tolerance
                    and relative_change <= args.attachment_relative_tolerance
                ):
                    stable_rounds += 1
                else:
                    stable_rounds = 0
            previous_attachment = current_attachment
            if (
                refinement_round >= args.minimum_refinement_rounds
                and stable_rounds >= args.convergence_patience
            ):
                break

    selected = best_final_feasible or best_feasible or best_infeasible
    if selected is None:
        raise RuntimeError("Optimization did not produce a finite candidate")
    best_momentum = selected["momentum"]

    validation_steps = args.steps * 2
    normalized_template = (template["points"] - origin) / scale
    deformed = (
        shoot_chunks(torch, control, best_momentum, normalized_template, 1.0, validation_steps)
        * scale
        + origin
    )
    plane_error = float(
        np.max(
            np.abs(
                np.einsum(
                    "ij,j->i",
                    deformed[template["basal_ids"]],
                    target["plane_normal"],
                )
                - target["plane_offset"]
            )
        )
    )
    plane_residuals = (
        np.einsum(
            "ij,j->i", deformed[template["basal_ids"]], target["plane_normal"]
        )
        - target["plane_offset"]
    )
    qa = surface_qa(template["points"], deformed, template["faces"])
    full_distance = symmetric_surface_metrics(
        deformed, template["faces"], target["points"], target["faces"]
    )
    anatomical_distance = symmetric_surface_metrics(
        deformed,
        template["faces"][~template["basal_faces"].astype(bool)],
        target["points"],
        target["faces"][~target["basal_faces"].astype(bool)],
    )
    np.savez_compressed(
        args.output_dir / "momenta.npz",
        control_points=control_points,
        momenta=best_momentum.cpu().numpy() * scale,
        origin=origin,
        scale=scale,
        steps=validation_steps,
        plane_normal=target["plane_normal"],
        plane_offset=target["plane_offset"],
        template=template_name,
        target=args.target,
        registration_version="multiobject_endpoint_constraint_v2",
    )
    write_surface(
        args.output_dir / "registered_surface.vtk",
        deformed,
        template["faces"],
        template["basal_faces"],
        template["object_labels"],
    )
    saved = np.load(args.output_dir / "momenta.npz")
    replayed = (
        shoot_chunks(
            torch,
            torch.tensor((saved["control_points"] - saved["origin"]) / float(saved["scale"])),
            torch.tensor(saved["momenta"] / float(saved["scale"])),
            normalized_template,
            1.0,
            int(saved["steps"]),
        )
        * scale
        + origin
    )
    replay_error = float(np.max(np.abs(replayed - deformed)))
    passed = (
        plane_error <= args.flat_tolerance
        and anatomical_distance["mean"] <= args.maximum_mean_distance
        and anatomical_distance["p95"] <= args.maximum_p95_distance
        and replay_error <= 1e-10
        and qa["degenerate_triangles"] == 0
        and qa["minimum_triangle_area_ratio"] >= args.minimum_area_ratio
    )
    save_json(
        args.output_dir / "report.json",
        {
            "status": "registration_checks_passed" if passed else "REJECTED",
            "template": template_name,
            "target": args.target,
            "output_points": int(len(deformed)),
            "output_triangles": int(len(template["faces"])),
            "common_control_points": int(len(control_points)),
            "initial_normalized_attachment": float(initial_attachment),
            "selected_candidate": {
                key: value for key, value in selected.items() if key != "momentum"
            },
            "maximum_plane_error": plane_error,
            "rms_plane_error": float(np.sqrt(np.mean(plane_residuals**2))),
            "flat_tolerance": args.flat_tolerance,
            "replay_maximum_coordinate_error": replay_error,
            "full_surface_distance": full_distance,
            "non_basal_anatomical_distance": anatomical_distance,
            "anatomical_distance_limits": {
                "maximum_mean": args.maximum_mean_distance,
                "maximum_p95": args.maximum_p95_distance,
            },
            "surface_qa": qa,
            "component_pairing_score": pairing_score,
            "component_pairs": pair_report,
            "history": history,
            "deformetrica_compatible": False,
            "limitations": [
                "No surface self-intersection test",
                "Automatic basal and component labels require visual review",
                "Endpoint flatness does not guarantee flatness for arbitrary VAE samples",
                "Distance limits are pilot values and should be tied to image resolution",
            ],
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "function"
            },
        },
    )
    print(
        f"Status: {'PASS' if passed else 'REJECTED'}; plane error={plane_error:.6g}; "
        f"anatomical mean/p95={anatomical_distance['mean']:.6g}/"
        f"{anatomical_distance['p95']:.6g}; replay error={replay_error:.3g}",
        flush=True,
    )


def replay(args):
    torch = torch_setup()
    saved = np.load(args.momenta)
    template = np.load(args.template_data)
    scale = float(saved["scale"])
    origin = saved["origin"]
    control = torch.tensor((saved["control_points"] - origin) / scale)
    momentum = torch.tensor(saved["momenta"] / scale)
    points = (template["points"] - origin) / scale
    result = shoot_chunks(torch, control, momentum, points, 1.0, int(saved["steps"]))
    result = result * scale + origin
    if args.output.exists():
        raise ValueError(f"Output exists: {args.output}")
    write_surface(
        args.output,
        result,
        template["faces"],
        template["basal_faces"],
        template["object_labels"] if "object_labels" in template.files else None,
    )
    error = np.max(
        np.abs(
            np.einsum(
                "ij,j->i", result[template["basal_ids"]], saved["plane_normal"]
            )
            - saved["plane_offset"]
        )
    )
    print(f"Maximum basal-plane error: {error:.9g}")


def publish_exact(source, destination):
    """Publish without silently replacing a different existing result."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise FileExistsError(
                f"Refusing to overwrite a different existing file: {destination}"
            )
        return "already present"
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"Published file failed checksum verification: {destination}")
    return "written"


def completed_run(run_dir, target):
    required = [
        run_dir / "registered_surface.vtk",
        run_dir / "momenta.npz",
        run_dir / "report.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    report = json.loads(required[-1].read_text())
    return (
        report.get("status") == "registration_checks_passed"
        and report.get("target") == target
    )


def register_all(args):
    metadata_dir = args.registered_dir / "RegistrationMetadata"
    prepared_dir = metadata_dir / "Prepared"
    runs_dir = metadata_dir / "Runs"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    prepared_now = False
    if not (prepared_dir / "manifest.json").is_file():
        if prepared_dir.exists():
            raise ValueError(
                f"Incomplete prepared directory exists: {prepared_dir}. "
                "Move it aside and rerun; it will not be overwritten."
            )
        prepare(
            SimpleNamespace(
                surface_dir=args.surface_dir,
                output_dir=prepared_dir,
                expected_count=args.expected_count,
                template=args.template,
                plane_tolerance=args.plane_tolerance,
                max_tilt=args.max_tilt,
                expected_objects=args.expected_objects,
            )
        )
        prepared_now = True

    if not args.reviewed_base_labels:
        message = (
            f"Review the label surfaces in {prepared_dir} in ParaView. "
            "When the BasalFace and AnatomicalObject labels are correct, rerun "
            "this command with --reviewed-base-labels."
        )
        if prepared_now:
            print(f"\nPreparation complete. {message}", flush=True)
            return
        raise ValueError(message)

    manifest = json.loads((prepared_dir / "manifest.json").read_text())
    available = [record["subject"] for record in manifest["subjects"]]
    targets = args.targets or available
    unknown = sorted(set(targets) - set(available))
    if unknown:
        raise ValueError(f"Unknown targets: {', '.join(unknown)}")

    summary = []
    for index, target in enumerate(targets, start=1):
        run_dir = runs_dir / target
        print(f"\n=== [{index}/{len(targets)}] {target} ===", flush=True)
        if run_dir.exists() and not completed_run(run_dir, target):
            raise ValueError(
                f"Incomplete or rejected run already exists: {run_dir}. "
                "Preserve it for diagnosis and move it aside before retrying."
            )
        if not run_dir.exists():
            registration_args = SimpleNamespace(**vars(args))
            registration_args.prepared_dir = prepared_dir
            registration_args.target = target
            registration_args.output_dir = run_dir
            register(registration_args)
        if not completed_run(run_dir, target):
            raise RuntimeError(f"Registration QA did not pass for {target}")

        case_number = target.removeprefix("ct_case_")
        surface_destination = (
            args.registered_dir / f"ct_case_registered_{case_number}.vtk"
        )
        momentum_destination = args.momenta_dir / f"{int(case_number):02d}_momenta.npz"
        surface_action = publish_exact(
            run_dir / "registered_surface.vtk", surface_destination
        )
        momentum_action = publish_exact(run_dir / "momenta.npz", momentum_destination)
        report = json.loads((run_dir / "report.json").read_text())
        summary.append(
            {
                "target": target,
                "registered_surface": str(surface_destination),
                "momenta": str(momentum_destination),
                "report": str(run_dir / "report.json"),
                "maximum_plane_error": report["maximum_plane_error"],
                "anatomical_mean_distance": report["non_basal_anatomical_distance"]["mean"],
                "anatomical_p95_distance": report["non_basal_anatomical_distance"]["p95"],
                "surface_publish_action": surface_action,
                "momenta_publish_action": momentum_action,
            }
        )
        save_json(metadata_dir / "batch_manifest.json", {"status": "in_progress", "cases": summary})

    save_json(
        metadata_dir / "batch_manifest.json",
        {
            "status": "complete",
            "template": args.template,
            "registered_surfaces": len(summary),
            "momentum_archives": len(summary),
            "cases": summary,
        },
    )
    print(
        f"\nComplete: {len(summary)} registered surfaces in {args.registered_dir}; "
        f"{len(summary)} momentum archives in {args.momenta_dir}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser(
        "all", help="prepare and register every CT surface to CT55"
    )
    sub.add_argument("--surface-dir", type=Path, default=Path("04.1_CTRegistrationSurfaces"))
    sub.add_argument("--registered-dir", type=Path, default=Path("05.1CTRegistered"))
    sub.add_argument("--momenta-dir", type=Path, default=Path("05.2_Moments"))
    sub.add_argument("--expected-count", type=int, default=61)
    sub.add_argument("--template", default="ct_case_0055")
    sub.add_argument("--targets", nargs="+", help="optional subset, e.g. ct_case_0001")
    sub.add_argument("--plane-tolerance", type=float, default=0.05)
    sub.add_argument("--max-tilt", type=float, default=40.0)
    sub.add_argument("--expected-objects", type=int, default=3)
    sub.add_argument("--reviewed-base-labels", action="store_true")
    sub.add_argument("--fitting-triangles", type=int, default=4000)
    sub.add_argument("--steps", type=int, default=8)
    sub.add_argument("--outer-iterations", type=int, default=8)
    sub.add_argument("--inner-iterations", type=int, default=30)
    sub.add_argument("--refinement-rounds", type=int, default=2)
    sub.add_argument("--minimum-refinement-rounds", type=int, default=2)
    sub.add_argument("--refinement-iterations", type=int, default=25)
    sub.add_argument("--convergence-patience", type=int, default=1)
    sub.add_argument("--kernel-width", type=float, default=16.0)
    sub.add_argument("--attachment-widths", type=float, nargs="+", default=[12.0, 8.0, 4.0])
    sub.add_argument("--contour-width", type=float, default=4.0)
    sub.add_argument("--contour-weight", type=float, default=0.25)
    sub.add_argument("--control-spacing", type=float, default=8.0)
    sub.add_argument("--energy-weight", type=float, default=0.01)
    sub.add_argument("--initial-penalty", type=float, default=10.0)
    sub.add_argument("--penalty-growth", type=float, default=10.0)
    sub.add_argument("--maximum-penalty", type=float, default=10000.0)
    sub.add_argument("--flat-tolerance", type=float, default=0.15)
    sub.add_argument("--optimization-flat-fraction", type=float, default=0.8)
    sub.add_argument("--attachment-relative-tolerance", type=float, default=1e-4)
    sub.add_argument("--learning-rate", type=float, default=0.5)
    sub.add_argument("--gradient-tolerance", type=float, default=1e-9)
    sub.add_argument("--parameter-tolerance", type=float, default=1e-12)
    sub.add_argument("--minimum-area-ratio", type=float, default=0.02)
    sub.add_argument("--maximum-mean-distance", type=float, default=2.0)
    sub.add_argument("--maximum-p95-distance", type=float, default=5.0)
    sub.set_defaults(function=register_all)

    sub = commands.add_parser("prepare")
    sub.add_argument("--surface-dir", type=Path, default=Path("CTRegistrationSurfaces"))
    sub.add_argument("--output-dir", type=Path, default=Path("CTMultiObjectPrepared"))
    sub.add_argument("--expected-count", type=int, default=61)
    sub.add_argument("--template", default="ct_case_0055")
    sub.add_argument("--plane-tolerance", type=float, default=0.05)
    sub.add_argument("--max-tilt", type=float, default=40.0)
    sub.add_argument("--expected-objects", type=int, default=3)
    sub.set_defaults(function=prepare)

    sub = commands.add_parser("register")
    sub.add_argument("--prepared-dir", type=Path, default=Path("CTMultiObjectPrepared"))
    sub.add_argument("--target", required=True)
    sub.add_argument("--output-dir", type=Path, required=True)
    sub.add_argument("--reviewed-base-labels", action="store_true")
    sub.add_argument("--fitting-triangles", type=int, default=4000)
    sub.add_argument("--steps", type=int, default=8)
    sub.add_argument("--outer-iterations", type=int, default=8)
    sub.add_argument("--inner-iterations", type=int, default=30)
    sub.add_argument("--refinement-rounds", type=int, default=2)
    sub.add_argument("--minimum-refinement-rounds", type=int, default=2)
    sub.add_argument("--refinement-iterations", type=int, default=25)
    sub.add_argument("--convergence-patience", type=int, default=1)
    sub.add_argument("--kernel-width", type=float, default=16.0)
    sub.add_argument(
        "--attachment-widths",
        "--attachment-width",
        dest="attachment_widths",
        type=float,
        nargs="+",
        default=[12.0, 8.0, 4.0],
        help="Coarse-to-fine surface attachment widths; the old singular flag is accepted",
    )
    sub.add_argument("--contour-width", type=float, default=4.0)
    sub.add_argument("--contour-weight", type=float, default=0.25)
    sub.add_argument("--control-spacing", type=float, default=8.0)
    sub.add_argument("--energy-weight", type=float, default=0.01)
    sub.add_argument("--initial-penalty", type=float, default=10.0)
    sub.add_argument("--penalty-growth", type=float, default=10.0)
    sub.add_argument("--maximum-penalty", type=float, default=10000.0)
    sub.add_argument("--flat-tolerance", type=float, default=0.15)
    sub.add_argument("--optimization-flat-fraction", type=float, default=0.8)
    sub.add_argument("--attachment-relative-tolerance", type=float, default=1e-4)
    sub.add_argument("--learning-rate", type=float, default=0.5)
    sub.add_argument("--gradient-tolerance", type=float, default=1e-9)
    sub.add_argument("--parameter-tolerance", type=float, default=1e-12)
    sub.add_argument("--minimum-area-ratio", type=float, default=0.02)
    sub.add_argument("--maximum-mean-distance", type=float, default=2.0)
    sub.add_argument("--maximum-p95-distance", type=float, default=5.0)
    sub.set_defaults(function=register)

    sub = commands.add_parser("replay")
    sub.add_argument("--momenta", type=Path, required=True)
    sub.add_argument("--template-data", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.set_defaults(function=replay)

    args = parser.parse_args()
    positive = (
        "expected_count",
        "plane_tolerance",
        "max_tilt",
        "expected_objects",
        "fitting_triangles",
        "steps",
        "outer_iterations",
        "inner_iterations",
        "refinement_rounds",
        "minimum_refinement_rounds",
        "refinement_iterations",
        "convergence_patience",
        "kernel_width",
        "contour_width",
        "contour_weight",
        "control_spacing",
        "energy_weight",
        "initial_penalty",
        "penalty_growth",
        "maximum_penalty",
        "flat_tolerance",
        "optimization_flat_fraction",
        "attachment_relative_tolerance",
        "learning_rate",
        "gradient_tolerance",
        "parameter_tolerance",
        "minimum_area_ratio",
        "maximum_mean_distance",
        "maximum_p95_distance",
    )
    for name in positive:
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "attachment_widths") and any(
        width <= 0 for width in args.attachment_widths
    ):
        parser.error("--attachment-widths values must be positive")
    if hasattr(args, "attachment_widths") and any(
        left < right
        for left, right in zip(args.attachment_widths, args.attachment_widths[1:])
    ):
        parser.error("--attachment-widths must be coarse-to-fine (non-increasing)")
    if hasattr(args, "optimization_flat_fraction") and not (
        0 < args.optimization_flat_fraction <= 1
    ):
        parser.error("--optimization-flat-fraction must be in (0, 1]")
    if hasattr(args, "minimum_refinement_rounds") and (
        args.minimum_refinement_rounds > args.refinement_rounds
    ):
        parser.error("--minimum-refinement-rounds cannot exceed --refinement-rounds")
    if hasattr(args, "maximum_penalty") and args.maximum_penalty < args.initial_penalty:
        parser.error("--maximum-penalty cannot be smaller than --initial-penalty")
    args.function(args)


if __name__ == "__main__":
    main()
