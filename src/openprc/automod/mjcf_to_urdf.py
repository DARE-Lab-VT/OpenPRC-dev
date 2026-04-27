"""
MJCF → URDF conversion helper.

Wraps the `mjcf_urdf_simple_converter` package for kinematics-preserving
conversion of MuJoCo MJCF files into URDF. Dynamics parameters (joint damping,
armature, solver settings) are MuJoCo-specific and not preserved by URDF —
but for visualization and forward kinematics, URDF is sufficient.

Why we do this:
  - URDF is the de facto interchange format for robotics tooling.
  - Most dataset joint-name conventions assume URDF names.
  - Our bundle consumers (PiViz, PRC training code) only need kinematics,
    not MuJoCo dynamics.

Usage:
    from mjcf_to_urdf import convert_mjcf_to_urdf
    convert_mjcf_to_urdf(
        mjcf_path="menagerie/unitree_go1/go1.xml",
        urdf_out="go1/urdf/go1.urdf",
        mesh_dir="go1/urdf/meshes",
    )

Requires: pip install mjcf_urdf_simple_converter
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


def convert_mjcf_to_urdf(
    mjcf_path: str,
    urdf_out: str,
    mesh_dir: Optional[str] = None,
    copy_meshes: bool = True,
) -> str:
    """
    Convert an MJCF file to URDF.

    Args:
        mjcf_path: Path to the source .xml (MJCF) file.
        urdf_out: Path where the output .urdf should be written.
        mesh_dir: Where to copy mesh files referenced by the MJCF.
                  If None, defaults to a `meshes/` subdir next to urdf_out.
        copy_meshes: If True, copy all referenced .obj/.stl files into
                     mesh_dir and rewrite URDF paths to reference them.

    Returns:
        Path to the output URDF.

    Raises:
        ImportError: if mjcf_urdf_simple_converter is not installed.
        FileNotFoundError: if mjcf_path doesn't exist.
    """
    try:
        from mjcf_urdf_simple_converter import convert
    except ImportError as e:
        raise ImportError(
            "mjcf_urdf_simple_converter is required for MJCF→URDF conversion. "
            "Install with: pip install mjcf_urdf_simple_converter\n"
            "Alternatively, use an upstream URDF directly (see the "
            "per-robot fetch scripts for examples)."
        ) from e

    mjcf_path = os.path.abspath(mjcf_path)
    urdf_out = os.path.abspath(urdf_out)

    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

    os.makedirs(os.path.dirname(urdf_out), exist_ok=True)
    if mesh_dir is None:
        mesh_dir = os.path.join(os.path.dirname(urdf_out), "meshes")
    mesh_dir = os.path.abspath(mesh_dir)
    os.makedirs(mesh_dir, exist_ok=True)

    # mjcf_urdf_simple_converter API (as of v0.2): convert(mjcf_file, urdf_file)
    convert(mjcf_path, urdf_out)

    if copy_meshes:
        _copy_meshes_and_rewrite(mjcf_path, urdf_out, mesh_dir)

    return urdf_out


def _copy_meshes_and_rewrite(mjcf_path: str, urdf_path: str, mesh_dir: str):
    """
    Ensure the URDF's mesh references resolve to files that actually exist
    relative to the URDF directory.

    Two converter behaviors need to be handled:

      1. `mjcf_urdf_simple_converter` writes OBJ/MTL pairs with color-baked
         names directly into `<urdf_dir>/meshes/` (e.g.
         `converted_left_hip_yaw_link_b2b2b2ff.obj`). The URDF already
         references them at that relative path. Nothing to copy — just verify
         they exist and leave the URDF alone.

      2. Other converters emit URDFs that reference the *original* Menagerie
         mesh files (e.g. `meshes/foot.stl`). Those files need to be copied
         from the MJCF's `assets/` directory into our mesh_dir, and the URDF
         paths rewritten if necessary.

    This function handles both cases: for each mesh reference, it checks if
    the file already resolves relative to the URDF directory. If yes, it's
    untouched. If no, it tries the Menagerie `assets/` paths and copies.
    """
    import re

    mjcf_dir = os.path.dirname(mjcf_path)
    mjcf_assets = os.path.join(mjcf_dir, "assets")
    urdf_dir = os.path.dirname(urdf_path)

    with open(urdf_path) as f:
        urdf_text = f.read()

    mesh_refs = re.findall(r'filename="([^"]+\.(?:stl|obj|STL|OBJ))"', urdf_text)
    mesh_refs = list(set(mesh_refs))

    copied = 0
    already_present = 0
    missing = []
    rewritten_urdf = urdf_text

    for ref in mesh_refs:
        basename = os.path.basename(ref)

        # CASE 1: URDF reference already resolves relative to URDF dir.
        # This is the common case when the converter wrote the meshes itself
        # (mjcf_urdf_simple_converter). Nothing to do.
        urdf_relative_path = os.path.join(urdf_dir, ref)
        if os.path.exists(urdf_relative_path):
            already_present += 1
            continue

        # CASE 2: URDF references meshes that live somewhere else. Try the
        # Menagerie asset conventions.
        candidates = [
            ref,                                    # absolute or relative to cwd
            os.path.join(mjcf_dir, ref),            # relative to MJCF
            os.path.join(mjcf_assets, basename),    # in assets/
            os.path.join(mjcf_dir, basename),       # next to MJCF
        ]
        src = next((p for p in candidates if os.path.exists(p)), None)
        if src is None:
            missing.append(ref)
            continue

        dst = os.path.join(mesh_dir, basename)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            copied += 1

        # Rewrite the URDF reference to point at our copy
        rel_mesh = os.path.relpath(dst, urdf_dir)
        rewritten_urdf = rewritten_urdf.replace(
            f'filename="{ref}"', f'filename="{rel_mesh}"'
        )

    if rewritten_urdf != urdf_text:
        with open(urdf_path, "w") as f:
            f.write(rewritten_urdf)

    print(f"  mesh references: {already_present} already in place, "
          f"{copied} copied from MJCF assets, {len(missing)} missing")
    if missing:
        print(f"  missing references (first 5): {missing[:5]}")
