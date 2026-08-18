"""
PiViz rendering adapter.

The one boundary between our FK engine's 4x4 world transforms and PiViz's
pose-based draw_mesh API. Kept tiny deliberately — if PiViz adds a matrix-input
API later we change it here and nowhere else.

Design notes:

  - We call `pgfx.draw_mesh` per link rather than `draw_meshes_batch` because
    every link has a *different* mesh file. `draw_meshes_batch` batches N
    instances of the *same* mesh — useful for particle fields, not for robots.
    For a Go1 with ~20 links we issue ~20 draw calls per frame; PiViz batches
    those internally via its automatic geometry cache, so this is not a perf
    problem in practice.

  - Mesh scale comes from the URDF's <mesh scale="..."/> attribute (usually
    [1,1,1]). We pass it through; PiViz accepts a (3,) scale vector.

  - URDF convention is Z-up. PiViz defaults to Z-up too (the camera 'iso'
    view is Z-up per the stress-test example). OBJ files are often authored
    Y-up, but Menagerie ships STLs authored in the URDF's native frame, so
    no per-mesh base rotation is needed. If we see axes looking wrong at
    animation time we add it here.

  - `mtl=''` is used so meshes render with their link's tint color rather
    than any embedded material. For the Go1 everything is dark gray anyway.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from urdf_fk import LinkVisual, decompose_pose


# Default color palette for robots — grayscale with slight variation per link
# so adjacent links are visually distinguishable. Index modulo palette length.
DEFAULT_LINK_COLORS = [
    (0.70, 0.70, 0.72, 1.0),  # pale gray
    (0.55, 0.55, 0.58, 1.0),  # medium gray
    (0.40, 0.40, 0.42, 1.0),  # dark gray
    (0.60, 0.55, 0.50, 1.0),  # warm gray
]


def draw_robot(
    pgfx,
    visuals: Sequence[LinkVisual],
    link_transforms: Dict[str, np.ndarray],
    link_colors: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> int:
    """
    Draw every visual of the robot for one frame.

    Args:
        pgfx: the PiViz primitives module (from `from piviz import pgfx`).
        visuals: list of LinkVisual from RobotFK.visuals.
        link_transforms: {link_name: (4,4)} from RobotFK.compute(...).
        link_colors: optional per-link override; any link not in this dict
                     falls back to DEFAULT_LINK_COLORS cycled by index.

    Returns:
        Number of draw calls issued (useful for perf HUDs).
    """
    n_calls = 0
    # Assign stable colors by sorted link-name order so the same link always
    # gets the same color across frames regardless of dict iteration order.
    ordered_links = sorted({v.link_name for v in visuals})
    default_map = {
        name: DEFAULT_LINK_COLORS[i % len(DEFAULT_LINK_COLORS)]
        for i, name in enumerate(ordered_links)
    }

    for v in visuals:
        if v.link_name not in link_transforms:
            continue  # should not happen for visuals we collected, but be safe
        T_world_link = link_transforms[v.link_name]
        # Compose the mesh-origin offset: mesh is placed at link_frame @ mesh_origin
        T_world_mesh = T_world_link @ v.mesh_origin

        pos, euler = decompose_pose(T_world_mesh)
        scale = tuple(float(s) for s in v.scale)

        color = (link_colors or {}).get(v.link_name, default_map[v.link_name])

        pgfx.draw_mesh(
            v.mesh_path,
            position=tuple(float(x) for x in pos),
            scale=scale,
            rotation=euler,
            color=color,
            mtl='',  # ignore any MTL; use our solid tint
        )
        n_calls += 1
    return n_calls
