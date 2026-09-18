#!/usr/bin/env python3
"""Collect CT VTU meshes into MixedCohort without modifying originals."""

from pathlib import Path
import shutil
import sys


SOURCE_ROOT = Path("CT_original")
OUTPUT_DIR = Path("MixedCohort")
SOURCE_RELATIVE_PATH = Path("05_mesh_ref/mesh-complete.mesh.vtu")


def main() -> int:
    if not SOURCE_ROOT.is_dir():
        print(f"ERROR: folder not found: {SOURCE_ROOT}", file=sys.stderr)
        return 1

    case_dirs = sorted(
        path for path in SOURCE_ROOT.glob("ct_case_*")
        if path.is_dir()
    )

    if not case_dirs:
        print(f"ERROR: no ct_case_* folders found in {SOURCE_ROOT}", file=sys.stderr)
        return 1

    # Inspect everything before creating or copying anything.
    copy_plan = []
    missing = []

    for case_dir in case_dirs:
        source = case_dir / SOURCE_RELATIVE_PATH
        destination = OUTPUT_DIR / f"{case_dir.name}.vtu"

        if source.is_file():
            copy_plan.append((source, destination))
        else:
            missing.append(source)

    print(f"Case folders found: {len(case_dirs)}")
    print(f"VTU files found:    {len(copy_plan)}")
    print(f"Missing VTU files:  {len(missing)}")

    if missing:
        print("\nMissing files:")
        for path in missing:
            print(f"  {path}")

        print("\nNothing was copied because the cohort is incomplete.")
        return 1

    existing = [
        destination for _, destination in copy_plan
        if destination.exists()
    ]

    if existing:
        print("\nERROR: destination files already exist:")
        for path in existing:
            print(f"  {path}")
        print("\nNothing was overwritten.")
        return 1

    print("\nCopy plan:")
    for source, destination in copy_plan:
        print(f"  {source} -> {destination}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for source, destination in copy_plan:
        # copy2 preserves modification-time metadata.
        shutil.copy2(source, destination)

    output_files = sorted(OUTPUT_DIR.glob("ct_case_*.vtu"))
    print(f"\nSuccessfully copied {len(output_files)} VTU files.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    if len(output_files) != len(copy_plan):
        print("ERROR: final file count does not match the copy plan.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())