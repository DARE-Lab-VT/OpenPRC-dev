"""
Mesh → spring-mass reservoir preprocessing.

Turns a robot link's visual mesh (.stl / .obj) into a per-link spring-mass
reservoir spec saved as an .npz. The reservoir consists of:

  - Anchor nodes: decimated surface vertices, first K in the node array.
    At runtime these are driven by DEMLAT position actuators that follow
    the link's FK transform — so they move rigidly with the robot link.
  - Interior nodes: Poisson-disk samples of the mesh volume, indices [K, K+N).
    These are free masses that respond to spring forces.
  - Springs: radius-based k-nearest-neighbor connectivity. Anchor↔interior
    and interior↔interior only. No anchor↔anchor springs (they'd be useless
    — both endpoints are hard-constrained).

The .npz is stored in link-local frame. Conversion to world frame happens at
runtime when the trajectory transforms arrive.

Usage:
    python mesh_to_reservoir.py \\
        path/to/mesh.stl \\
        path/to/output.npz \\
        [--preset small|medium|large]
        [--decimate-ratio 0.05] [--node-density 500000] [...]
        [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

@dataclass
class ReservoirParams:
    """All tunable parameters for reservoir generation.

    Saved as JSON inside the .npz so every file records exactly how it was
    produced — crucial for reproducibility and for sweeping parameter grids
    during the PRC training phase.
    """
    # Surface decimation → anchors
    decimate_ratio: float = 0.05          # fraction of surface vertices to keep
    min_anchors: int = 8                  # per-link minimum
    max_anchors: int = 200                # per-link maximum

    # Volume sampling → interior nodes
    # Default: ~5 nodes per cm³, which gives visually dense reservoirs for
    # typical robot links (1cm³ = 1 mL). A 300 cm³ calf gets ~1500 interior
    # nodes — plenty of reservoir state for PRC training.
    node_density: float = 5_000_000.0     # target interior nodes per m³
    min_nodes: int = 20                   # interior-node minimum
    max_nodes: int = 3000                 # interior-node maximum

    # Interior topology — how nodes are placed and connected
    # "bcc"     : body-centered cubic lattice clipped to mesh (clean, regular,
    #             every node has the same neighbor count, ideal for theory)
    # "poisson" : Poisson-disk volume sampling (random, denser packing but
    #             visually clumpy connectivity)
    interior_topology: str = "bcc"

    # Randomization (breaks grid/lattice artifacts, helps reservoir expressivity)
    jitter_fraction: float = 0.1          # position noise as multiple of avg spacing
    stiffness_jitter_fraction: float = 0.2  # log-normal stddev

    # Connectivity (used for poisson topology and for anchor↔interior links)
    connection_radius_factor: float = 3.0  # multiplier on avg node spacing
    max_neighbors: int = 12

    # Physics defaults (applied to every spring unless overridden at runtime)
    base_stiffness: float = 10.0
    base_damping: float = 0.1
    base_node_mass: float = 0.01

    # Reproducibility
    rng_seed: int = 0


PRESETS = {
    "small":  ReservoirParams(decimate_ratio=0.03, max_anchors=40,
                              node_density=1_000_000.0, max_nodes=500),
    "medium": ReservoirParams(),  # defaults: ~5M/m³, caps at 3000 nodes
    "large":  ReservoirParams(decimate_ratio=0.10, max_anchors=500,
                              node_density=20_000_000.0, max_nodes=10000),
}


# -----------------------------------------------------------------------------
# Pure-function core (no trimesh, fully testable with numpy only)
# -----------------------------------------------------------------------------

def avg_nearest_neighbor_spacing(positions: np.ndarray) -> float:
    """
    Average distance from each point to its nearest (non-self) neighbor.
    Used as a characteristic length scale for jitter and connection radius.
    """
    if len(positions) < 2:
        return 1.0  # degenerate, prevents divide-by-zero downstream
    # O(N²) — fine for N < 1000 per link, which is our regime.
    dists = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    np.fill_diagonal(dists, np.inf)
    return float(np.mean(dists.min(axis=1)))


def jitter_positions(
    positions: np.ndarray,
    jitter_fraction: float,
    avg_spacing: float,
    rng: np.random.Generator,
    lock_first: int = 0,
) -> np.ndarray:
    """
    Add isotropic Gaussian noise to positions.

    Args:
        positions: (N, 3)
        jitter_fraction: std dev as a multiple of avg_spacing
        avg_spacing: characteristic length scale
        rng: numpy Generator
        lock_first: if >0, the first `lock_first` positions are NOT jittered
                    (used to keep anchors exactly on the decimated surface).
    """
    if jitter_fraction <= 0.0:
        return positions.copy()
    noise = rng.normal(
        loc=0.0,
        scale=jitter_fraction * avg_spacing,
        size=positions.shape,
    ).astype(positions.dtype)
    if lock_first > 0:
        noise[:lock_first] = 0.0
    return positions + noise


def randomize_stiffness(
    base_stiffness: float,
    n_edges: int,
    jitter_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Per-spring stiffness = base × exp(jitter_fraction × N(0,1)).
    Log-normal keeps values positive.
    """
    if jitter_fraction <= 0.0:
        return np.full(n_edges, base_stiffness, dtype=np.float32)
    log_noise = rng.normal(0.0, jitter_fraction, size=n_edges)
    return (base_stiffness * np.exp(log_noise)).astype(np.float32)


def build_radius_knn_connectivity(
    positions: np.ndarray,
    radius: float,
    max_neighbors: int,
    anchor_count: int,
    skip_anchor_to_anchor: bool = True,
) -> np.ndarray:
    """
    For each node, connect it to up to max_neighbors within radius, prefering
    the closest. Builds a symmetric undirected graph as a unique (E, 2)
    edge list.

    Convention: node indices [0, anchor_count) are anchors, [anchor_count, N)
    are interior. If skip_anchor_to_anchor=True (default), anchor↔anchor pairs
    are dropped — they'd be useless since both endpoints are position-actuated.

    Returns:
        edges: (E, 2) int32 array with edges[e] = (i, j) where i < j.
    """
    n = positions.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int32)

    edges = set()

    # O(N²) distances — fine for our size. Vectorized so fast in practice.
    dists = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    np.fill_diagonal(dists, np.inf)

    for i in range(n):
        row = dists[i]
        # Indices within radius, sorted by distance ascending
        within = np.where(row <= radius)[0]
        if len(within) == 0:
            continue
        within_sorted = within[np.argsort(row[within])]
        # Keep up to max_neighbors
        take = within_sorted[:max_neighbors]
        for j in take:
            a, b = int(i), int(j)
            if a == b:
                continue
            if skip_anchor_to_anchor and a < anchor_count and b < anchor_count:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            edges.add((lo, hi))

    if not edges:
        return np.zeros((0, 2), dtype=np.int32)

    edge_arr = np.array(sorted(edges), dtype=np.int32)
    return edge_arr


def connected_components(n_nodes: int, edges: np.ndarray) -> list[set[int]]:
    """
    Compute connected components via union-find. Returns a list of sets of
    node indices. Useful for sanity-checking that the reservoir graph isn't
    fragmented.
    """
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in edges.tolist():
        union(int(a), int(b))

    buckets: dict[int, set[int]] = {}
    for i in range(n_nodes):
        r = find(i)
        buckets.setdefault(r, set()).add(i)
    return list(buckets.values())


def compute_rest_lengths(
    positions: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Spring rest lengths = Euclidean distance at rest."""
    if len(edges) == 0:
        return np.zeros(0, dtype=np.float32)
    diffs = positions[edges[:, 1]] - positions[edges[:, 0]]
    return np.linalg.norm(diffs, axis=1).astype(np.float32)


def bridge_components(
    positions: np.ndarray,
    edges: np.ndarray,
    anchor_count: int,
    skip_anchor_to_anchor: bool = True,
) -> np.ndarray:
    """
    Ensure the reservoir graph is fully connected.

    Fast path (typical): the interior nodes form a connected component (they're
    a BCC lattice or similar). Most disconnected components are isolated
    anchors with no interior in their connection radius. For each such anchor,
    link to its nearest interior node using a KDTree query — O(K log N).

    Fallback path: if even after the per-anchor fix the graph is still split
    (rare, only happens when the interior itself is fragmented), iteratively
    join the remaining components by finding the nearest pair across them via
    centroid-keyed KDTree.

    The legacy O(N²) per-pair bridging was correct but unscalable for thousands
    of components.
    """
    n = len(positions)
    comps = connected_components(n, edges)
    if len(comps) <= 1:
        return edges

    edges_set = {(int(a), int(b)) for a, b in edges}

    # Build node → component-id mapping
    node_comp = np.full(n, -1, dtype=np.int64)
    for ci, comp in enumerate(comps):
        for node_idx in comp:
            node_comp[node_idx] = ci

    interior_indices = np.arange(anchor_count, n)
    if len(interior_indices) > 0:
        # KDTree over interior nodes
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(positions[interior_indices])

            for a in range(anchor_count):
                # Skip anchors already connected to interior
                # Quick check: any neighbor of a is an interior node?
                connected_to_interior = False
                # Iterate edges_set is fast for small anchor degree;
                # but simpler: precompute anchor's component, check if it
                # contains any interior node.
                if any(idx >= anchor_count for idx in comps[node_comp[a]]):
                    connected_to_interior = True

                if connected_to_interior:
                    continue

                # Query nearest interior node to this anchor
                dist, j_local = tree.query(positions[a], k=1)
                j_global = int(interior_indices[j_local])
                lo, hi = (a, j_global) if a < j_global else (j_global, a)
                edges_set.add((lo, hi))

        except ImportError:
            # No scipy — fall back to numpy O(K * N_interior)
            int_pos = positions[interior_indices]
            for a in range(anchor_count):
                if any(idx >= anchor_count for idx in comps[node_comp[a]]):
                    continue
                d = np.linalg.norm(int_pos - positions[a], axis=1)
                j_local = int(np.argmin(d))
                j_global = int(interior_indices[j_local])
                lo, hi = (a, j_global) if a < j_global else (j_global, a)
                edges_set.add((lo, hi))

    # Recompute components — may still be split if interior itself fragments
    new_edges = np.array(sorted(edges_set), dtype=np.int32) \
        if edges_set else np.zeros((0, 2), dtype=np.int32)
    comps = connected_components(n, new_edges)
    if len(comps) <= 1:
        return new_edges

    # Fallback: bridge remaining components by closest-pair across centroids.
    # This loop runs at most len(comps)-1 times and each iteration is O(C log C)
    # with KDTree — much better than O(C²) brute force.
    while len(comps) > 1:
        try:
            from scipy.spatial import cKDTree
            centroids = np.array([positions[list(c)].mean(axis=0) for c in comps])
            ctree = cKDTree(centroids)
            # For each centroid, find its nearest *other* centroid (k=2 gets
            # self + nearest other)
            dists, idxs = ctree.query(centroids, k=2)
            # Pick the globally-shortest centroid pair to merge
            best_pair_idx = int(np.argmin(dists[:, 1]))
            i = best_pair_idx
            j = int(idxs[best_pair_idx, 1])
        except ImportError:
            # Brute force across centroids only (much smaller than full O(N²))
            centroids = np.array([positions[list(c)].mean(axis=0) for c in comps])
            d = np.linalg.norm(
                centroids[:, None, :] - centroids[None, :, :], axis=-1
            )
            np.fill_diagonal(d, np.inf)
            ij = np.unravel_index(np.argmin(d), d.shape)
            i, j = int(ij[0]), int(ij[1])

        # Find the actual nearest pair of nodes between comps[i] and comps[j]
        ci = list(comps[i])
        cj = list(comps[j])
        pi = positions[ci]
        pj = positions[cj]

        if skip_anchor_to_anchor:
            mask_i = np.array([idx < anchor_count for idx in ci])
            mask_j = np.array([idx < anchor_count for idx in cj])
            anchor_anchor = mask_i[:, None] & mask_j[None, :]
            d_pair = np.linalg.norm(pi[:, None, :] - pj[None, :, :], axis=-1)
            d_legal = np.where(anchor_anchor, np.inf, d_pair)
            if not np.any(np.isfinite(d_legal)):
                d_legal = d_pair  # fallback: allow anchor-anchor
        else:
            d_legal = np.linalg.norm(pi[:, None, :] - pj[None, :, :], axis=-1)

        loc = np.unravel_index(np.argmin(d_legal), d_legal.shape)
        a, b = ci[loc[0]], cj[loc[1]]
        lo, hi = (a, b) if a < b else (b, a)
        edges_set.add((lo, hi))

        # Merge components
        merged = comps[i] | comps[j]
        comps = [c for k, c in enumerate(comps) if k not in (i, j)]
        comps.append(merged)

    return np.array(sorted(edges_set), dtype=np.int32)


def bcc_lattice_in_bbox(
    bbox_min: np.ndarray, bbox_max: np.ndarray, spacing: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a body-centered cubic (BCC) point lattice covering an AABB.

    BCC = simple cubic at (i, j, k) plus an offset cubic at (i+0.5, j+0.5, k+0.5),
    both at the given spacing. Each interior point has 8 nearest neighbors at
    distance spacing*sqrt(3)/2 plus 6 second-nearest at distance `spacing`.

    Returns:
        positions: (N, 3) float64 lattice point coordinates
        is_offset: (N,) bool mask, True for points on the body-centered sublattice
                   (useful for diagnostics; not used downstream)
    """
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")

    # Pad the bbox slightly so we don't lose surface-adjacent points
    pad = spacing * 0.05
    lo = bbox_min - pad
    hi = bbox_max + pad

    # Corner sublattice (i, j, k) * spacing
    nx = int(np.ceil((hi[0] - lo[0]) / spacing)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / spacing)) + 1
    nz = int(np.ceil((hi[2] - lo[2]) / spacing)) + 1

    ix, iy, iz = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    corners = np.stack([
        lo[0] + ix.ravel() * spacing,
        lo[1] + iy.ravel() * spacing,
        lo[2] + iz.ravel() * spacing,
    ], axis=1)

    # Body-centered sublattice (i+0.5, j+0.5, k+0.5) * spacing
    bx, by, bz = np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
    )
    bodies = np.stack([
        lo[0] + (bx.ravel() + 0.5) * spacing,
        lo[1] + (by.ravel() + 0.5) * spacing,
        lo[2] + (bz.ravel() + 0.5) * spacing,
    ], axis=1)

    positions = np.vstack([corners, bodies])
    is_offset = np.concatenate([
        np.zeros(len(corners), dtype=bool),
        np.ones(len(bodies), dtype=bool),
    ])
    return positions, is_offset


def bcc_lattice_edges(
    positions: np.ndarray, spacing: float, tol_factor: float = 1.05
) -> np.ndarray:
    """
    Build canonical BCC edges using a distance-based test:
      - Body-centered ↔ corner: nearest-neighbor at distance spacing*sqrt(3)/2
      - Corner ↔ corner along axis: distance spacing

    Both NN distances are within tol_factor*spacing, so a single radius query
    catches them. tol_factor=1.05 accommodates floating-point drift.
    """
    n = len(positions)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int32)

    radius2 = (spacing * tol_factor) ** 2
    edges = set()
    diffs = positions[:, None, :] - positions[None, :, :]
    sq_dists = np.einsum("ijk,ijk->ij", diffs, diffs)
    np.fill_diagonal(sq_dists, np.inf)

    for i in range(n):
        within = np.where(sq_dists[i] <= radius2)[0]
        for j in within:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((int(a), int(b)))

    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(sorted(edges), dtype=np.int32)


# -----------------------------------------------------------------------------
# Mesh I/O (trimesh-dependent, not sandbox-testable)
# -----------------------------------------------------------------------------

def _lazy_import_trimesh():
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "trimesh is required for mesh preprocessing. "
            "Install with: pip install trimesh"
        ) from e
    return trimesh


def load_and_repair_mesh(mesh_path: str):
    """
    Load mesh, attempt to make watertight, fall back to convex hull if all
    repair strategies fail.

    Repair strategies tried in order:
      1. fill_holes + fix_normals (trimesh's built-in)
      2. merge coincident vertices + fill_holes (often needed for STLs)
      3. process=True during load (aggressive cleanup)
      4. Convex hull (last resort — warn user)

    Prints volume before/after repair so you can see if the fallback to
    convex hull happened and by how much the effective volume changed.
    """
    trimesh = _lazy_import_trimesh()
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh, got {type(mesh)} from {mesh_path}")

    orig_volume = mesh.volume if mesh.is_watertight else None

    # Strategy 1: basic fill_holes + fix_normals
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    # Strategy 2: merge very close vertices (STLs often have duplicates that
    # break manifoldness) and try fill_holes again
    if not mesh.is_watertight:
        try:
            mesh.merge_vertices()
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    # Strategy 3: re-load with aggressive processing
    if not mesh.is_watertight:
        try:
            reloaded = trimesh.load(mesh_path, force='mesh', process=True)
            if isinstance(reloaded, trimesh.Trimesh) and reloaded.is_watertight:
                mesh = reloaded
        except Exception:
            pass

    if mesh.is_watertight:
        print(f"  mesh: {len(mesh.vertices)} verts, "
              f"volume={mesh.volume:.6f} m³ (watertight)")
        return mesh

    # All repair attempts failed — convex hull
    hull = mesh.convex_hull
    print(f"  warning: mesh not watertight after repair; using convex hull")
    print(f"    hull volume: {hull.volume:.6f} m³ "
          f"(may be much larger than real mesh volume; reservoir will spill "
          f"outside the actual mesh shape)")
    return hull


def decimate_surface(mesh, decimate_ratio: float, min_anchors: int, max_anchors: int):
    """
    Decimate mesh surface to approximately (n_verts × decimate_ratio) vertices,
    clamped to [min_anchors, max_anchors]. Returns the decimated Trimesh.

    Handles multiple trimesh API versions:
      - Older: mesh.simplify_quadric_decimation(target_face_count)
      - Newer: mesh.simplify_quadric_decimation(target_reduction=fraction_to_remove)
               where fraction_to_remove is in (0, 1)
      - Both spellings: simplify_quadric_decimation / simplify_quadratic_decimation

    fast_simplification has a practical floor on how aggressively it can
    decimate (it stops when further reduction would destroy topology). So
    after quadric decimation we run a final FPS subsample to enforce the
    max_anchors cap exactly.
    """
    trimesh = _lazy_import_trimesh()
    n_orig_verts = len(mesh.vertices)
    n_orig_faces = len(mesh.faces)
    target_verts = int(round(n_orig_verts * decimate_ratio))
    target_verts = max(min_anchors, min(max_anchors, target_verts))

    # For a closed surface: F ≈ 2V (Euler). Target face count.
    target_faces = max(4, 2 * target_verts)
    target_reduction = max(0.001, min(0.999, 1.0 - target_faces / max(n_orig_faces, 1)))

    # Method name varies across trimesh versions
    if hasattr(mesh, 'simplify_quadric_decimation'):
        method = mesh.simplify_quadric_decimation
    elif hasattr(mesh, 'simplify_quadratic_decimation'):
        method = mesh.simplify_quadratic_decimation
    else:
        print("  warning: no quadric decimation method on this trimesh version; "
              "falling back to surface FPS")
        return _fps_fallback_decimation(mesh, target_verts)

    # Try both calling conventions in order of preference. Each trimesh version
    # accepts one but not the other; we don't know which without trying.
    decimated = None
    attempts = [
        ("target_reduction kwarg",
         lambda: method(target_reduction=target_reduction)),
        ("fractional positional",
         lambda: method(target_reduction)),
        ("target_faces kwarg",
         lambda: method(face_count=target_faces)),
        ("face count positional",
         lambda: method(target_faces)),
    ]
    last_error = None
    for label, attempt in attempts:
        try:
            decimated = attempt()
            if decimated is not None and len(decimated.vertices) > 0:
                break
            else:
                last_error = f"{label} produced empty mesh"
        except Exception as e:
            last_error = f"{label} failed: {e}"
            continue

    if decimated is None or len(decimated.vertices) == 0:
        print(f"  warning: all decimation attempts failed ({last_error}); "
              f"falling back to surface FPS")
        decimated = _fps_fallback_decimation(mesh, target_verts)

    # Hard cap: if quadric decimation couldn't reach the target (often the
    # case when the mesh has fine geometric features that resist aggressive
    # simplification), enforce max_anchors exactly via FPS on the decimated
    # vertices.
    if len(decimated.vertices) > max_anchors:
        n_before = len(decimated.vertices)
        decimated = _fps_subsample_vertices(decimated, max_anchors)
        n_after = len(decimated.vertices)
        if n_before != n_after:
            print(f"  decimated vertices clamped: {n_before} → {n_after} "
                  f"via FPS (max_anchors={max_anchors})")

    return decimated


def _fps_subsample_vertices(mesh, target_count: int):
    """
    Reduce a mesh to exactly `target_count` vertices via farthest-point
    sampling on its existing vertices. Returns a Trimesh with no faces (we
    only use the vertex positions downstream).
    """
    trimesh = _lazy_import_trimesh()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    n = len(verts)
    if target_count >= n:
        return mesh

    chosen = [0]
    dists_to_set = np.linalg.norm(verts - verts[0], axis=1)
    for _ in range(target_count - 1):
        next_idx = int(np.argmax(dists_to_set))
        chosen.append(next_idx)
        d_new = np.linalg.norm(verts - verts[next_idx], axis=1)
        dists_to_set = np.minimum(dists_to_set, d_new)
    return trimesh.Trimesh(
        vertices=verts[chosen],
        faces=np.zeros((0, 3), dtype=np.int64),
    )


def _fps_fallback_decimation(mesh, target_vertices: int):
    """
    Fallback for when proper surface decimation isn't available (e.g. the
    `fast_simplification` backend isn't installed).

    Strategy: oversample the mesh SURFACE uniformly by area, then run
    farthest-point sampling to pick `target_vertices` well-spread points.
    Produces better anchor distribution than subsampling the original vertex
    list, which can be clumpy on STLs with non-uniform tessellation.

    Returns a Trimesh carrying just the sampled points as vertices (no valid
    face list — caller only uses .vertices).
    """
    trimesh = _lazy_import_trimesh()

    # Oversample by 10× to give FPS enough candidates
    n_candidates = max(10 * target_vertices, 2000)
    try:
        candidates, _ = trimesh.sample.sample_surface(mesh, n_candidates)
    except Exception:
        # If surface sampling fails, fall back to vertex subsampling.
        candidates = np.asarray(mesh.vertices)

    candidates = np.asarray(candidates, dtype=np.float64)
    if target_vertices >= len(candidates):
        sampled_verts = candidates
    else:
        # Farthest-point sampling
        chosen = [0]
        dists_to_set = np.linalg.norm(candidates - candidates[0], axis=1)
        for _ in range(target_vertices - 1):
            next_idx = int(np.argmax(dists_to_set))
            chosen.append(next_idx)
            d_new = np.linalg.norm(candidates - candidates[next_idx], axis=1)
            dists_to_set = np.minimum(dists_to_set, d_new)
        sampled_verts = candidates[chosen]

    return trimesh.Trimesh(
        vertices=sampled_verts,
        faces=np.zeros((0, 3), dtype=np.int64),
    )


def sample_volume_interior(
    mesh,
    target_nodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Poisson-disk-ish sample of the mesh interior via rejection sampling.
    Returns (N, 3) float64 array of points inside the mesh.

    trimesh.sample.volume_mesh handles rejection. We oversample and trim.
    """
    if target_nodes <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    trimesh = _lazy_import_trimesh()
    # Seed via numpy rng to keep runs reproducible. trimesh uses its own
    # internals but is deterministic given the seed via np.random state.
    np.random.seed(int(rng.integers(0, 2**31 - 1)))

    # volume_mesh returns approximately `count` points, can be fewer if the
    # mesh volume is very small or sampling fails. Ask for a bit more.
    samples = trimesh.sample.volume_mesh(mesh, count=int(target_nodes * 1.2))

    if len(samples) < target_nodes:
        # Top up with surface sampling as a last resort
        surface_pts, _ = trimesh.sample.sample_surface(
            mesh, target_nodes - len(samples)
        )
        samples = np.vstack([samples, surface_pts])
    else:
        samples = samples[:target_nodes]

    return np.asarray(samples, dtype=np.float64)


def bcc_lattice_inside_mesh(
    mesh, target_nodes: int, min_nodes: int, max_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a body-centered cubic lattice covering the mesh's bbox, then
    keep only points strictly inside the mesh. Returns positions, the
    canonical BCC edge list (already pruned to in-mesh), and the lattice
    spacing used.

    Spacing is auto-tuned to land near `target_nodes` after clipping. We
    iterate at most a few times because mesh.volume relative to bbox volume
    determines clip rate, and we can predict it.
    """
    bbox_min = np.asarray(mesh.bounds[0], dtype=np.float64)
    bbox_max = np.asarray(mesh.bounds[1], dtype=np.float64)
    bbox_volume = float(np.prod(bbox_max - bbox_min))
    mesh_volume = float(mesh.volume)

    if bbox_volume <= 0 or mesh_volume <= 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32), 1.0

    # BCC lattice has 2 nodes per cubic cell of side `spacing`. Clipping
    # retains roughly (mesh_volume / bbox_volume) of them. So:
    #   target_nodes ≈ 2 * (bbox_volume / spacing^3) * (mesh_volume / bbox_volume)
    #   spacing^3   ≈ 2 * mesh_volume / target_nodes
    # Iterate twice to refine because mesh.contains can have edge effects.
    desired = max(min_nodes, min(max_nodes, target_nodes))
    spacing = (2.0 * mesh_volume / max(desired, 1)) ** (1.0 / 3.0)

    for _ in range(3):
        positions, _ = bcc_lattice_in_bbox(bbox_min, bbox_max, spacing)
        # mesh.contains is fast — vectorized ray casting
        inside = mesh.contains(positions)
        n_in = int(inside.sum())
        if n_in == 0:
            # Spacing too large; shrink
            spacing *= 0.5
            continue
        # If we're within ±20% of target, accept
        if 0.8 * desired <= n_in <= 1.25 * desired:
            break
        # Adjust spacing using cube-root scaling
        spacing *= (n_in / desired) ** (1.0 / 3.0)

    # Final pass with the converged spacing
    positions, _ = bcc_lattice_in_bbox(bbox_min, bbox_max, spacing)
    inside_mask = mesh.contains(positions)
    inside_positions = positions[inside_mask]

    if len(inside_positions) > max_nodes:
        # Hard cap: random subsample (preserves uniformity since the lattice was uniform)
        sel = np.random.default_rng(0).choice(
            len(inside_positions), max_nodes, replace=False
        )
        sel.sort()
        inside_positions = inside_positions[sel]

    # Build BCC edges on the clipped point set, using the same spacing
    edges = bcc_lattice_edges(inside_positions, spacing)
    return inside_positions, edges, spacing


# -----------------------------------------------------------------------------
# Top-level conversion
# -----------------------------------------------------------------------------

def mesh_to_reservoir(
    mesh_path: str,
    params: ReservoirParams,
    link_name: str = "",
    mesh_origin_in_link: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """
    Full preprocessing: mesh file → reservoir dict ready to save.

    Returns a dict with keys matching the .npz spec. Caller passes this to
    save_reservoir() to write the .npz file.
    """
    rng = np.random.default_rng(params.rng_seed)
    vprint = print if verbose else (lambda *a, **k: None)

    vprint(f"[{link_name or os.path.basename(mesh_path)}]")

    # 1. Load + repair (prints its own volume/watertight status)
    mesh = load_and_repair_mesh(mesh_path)

    # 2. Decimate surface → anchor positions
    decimated = decimate_surface(
        mesh, params.decimate_ratio, params.min_anchors, params.max_anchors
    )
    anchor_positions = np.asarray(decimated.vertices, dtype=np.float64)
    K = len(anchor_positions)
    vprint(f"  anchors: {K} (from {len(mesh.vertices)} verts, "
           f"ratio={K/max(1,len(mesh.vertices)):.2%})")

    # 3. Interior nodes — placement depends on topology choice
    target_interior = int(round(params.node_density * max(mesh.volume, 0.0)))
    target_interior = max(params.min_nodes, min(params.max_nodes, target_interior))

    interior_edges_local: np.ndarray  # edges among interior nodes only (local indexing)
    interior_edges_local = np.zeros((0, 2), dtype=np.int32)

    if params.interior_topology == "bcc":
        interior_positions, interior_edges_local, lattice_spacing = bcc_lattice_inside_mesh(
            mesh, target_interior, params.min_nodes, params.max_nodes,
        )
        N_interior = len(interior_positions)
        vprint(f"  interior: {N_interior} BCC lattice nodes "
               f"(spacing={lattice_spacing*1000:.2f} mm, target={target_interior})")
        vprint(f"  interior edges (BCC lattice): {len(interior_edges_local)}")
    elif params.interior_topology == "poisson":
        interior_positions = sample_volume_interior(mesh, target_interior, rng)
        N_interior = len(interior_positions)
        vprint(f"  interior: {N_interior} Poisson-disk samples "
               f"(density={params.node_density:.0e} /m³, target={target_interior})")
    else:
        raise ValueError(
            f"Unknown interior_topology: {params.interior_topology!r}. "
            f"Expected 'bcc' or 'poisson'."
        )

    # 4. Stack: anchors first, interior after
    node_positions = np.vstack([anchor_positions, interior_positions]).astype(np.float32)
    n_total = len(node_positions)

    # 5. Jitter positions (anchors locked at decimated surface)
    avg_spacing = avg_nearest_neighbor_spacing(node_positions)
    node_positions = jitter_positions(
        node_positions, params.jitter_fraction, avg_spacing, rng, lock_first=K
    )
    vprint(f"  avg node spacing: {avg_spacing*1000:.2f} mm")

    # 6. Connectivity
    if params.interior_topology == "bcc":
        # Interior↔interior edges are already given by the lattice. We only need
        # to add anchor↔interior edges via radius-knn (anchor↔anchor remains banned).
        # Shift interior_edges_local indices to global (anchor-prefixed) indexing.
        interior_edges_global = interior_edges_local + K  # all interior idx shift up by K

        # Build a graph with ONLY anchor↔interior connections via radius-knn,
        # by temporarily marking all interior nodes as "anchors" so the
        # connectivity helper skips interior↔interior — which we already have.
        # Cleaner: just call build_radius_knn_connectivity normally and union.
        anchor_radius = params.connection_radius_factor * avg_spacing
        anchor_iface_edges = build_radius_knn_connectivity(
            node_positions,
            radius=anchor_radius,
            max_neighbors=params.max_neighbors,
            anchor_count=K,
            skip_anchor_to_anchor=True,
        )
        # Filter to keep only anchor↔interior pairs (drop any interior↔interior
        # we'd duplicate from the BCC lattice)
        if len(anchor_iface_edges) > 0:
            mask = (anchor_iface_edges[:, 0] < K) ^ (anchor_iface_edges[:, 1] < K)
            anchor_iface_edges = anchor_iface_edges[mask]
        edges = np.vstack([interior_edges_global, anchor_iface_edges]) \
            if len(interior_edges_global) > 0 else anchor_iface_edges

        # Sort + dedupe
        if len(edges) > 0:
            edges_set = {(int(a), int(b)) for a, b in edges}
            edges = np.array(sorted(edges_set), dtype=np.int32)
        n_edges_initial = len(edges)
        vprint(f"  edges: {len(interior_edges_global)} interior (BCC) "
               f"+ {len(anchor_iface_edges)} anchor↔interior "
               f"= {n_edges_initial} total")
    else:
        # Poisson topology: single radius-knn pass over all nodes
        connection_radius = params.connection_radius_factor * avg_spacing
        edges = build_radius_knn_connectivity(
            node_positions,
            radius=connection_radius,
            max_neighbors=params.max_neighbors,
            anchor_count=K,
            skip_anchor_to_anchor=True,
        )
        n_edges_initial = len(edges)
        vprint(f"  edges (radius-knn): {n_edges_initial} "
               f"(radius={connection_radius*1000:.2f} mm)")

    # 7. Bridge any disconnected components
    components_before = connected_components(n_total, edges)
    if len(components_before) > 1:
        vprint(f"  bridging {len(components_before)} components...")
        edges = bridge_components(
            node_positions, edges, anchor_count=K, skip_anchor_to_anchor=True
        )
        n_bridges = len(edges) - n_edges_initial
        components_after = connected_components(n_total, edges)
        vprint(f"  added {n_bridges} bridge edges "
               f"({len(components_before)} → {len(components_after)} components)")
    n_edges = len(edges)
    vprint(f"  edges (final): {n_edges}")

    # 8. Rest lengths + stiffnesses + damping
    rest_lengths = compute_rest_lengths(node_positions, edges)
    stiffnesses = randomize_stiffness(
        params.base_stiffness, n_edges, params.stiffness_jitter_fraction, rng
    )
    dampings = np.full(n_edges, params.base_damping, dtype=np.float32)

    if mesh_origin_in_link is None:
        mesh_origin_in_link = np.eye(4, dtype=np.float32)

    return {
        "node_positions": node_positions,
        "n_anchors": K,
        "edges": edges,
        "rest_lengths": rest_lengths,
        "stiffnesses": stiffnesses,
        "damping_coefficients": dampings,
        "mesh_origin_in_link": np.asarray(mesh_origin_in_link, dtype=np.float32),
        "link_name": link_name,
        "rng_seed": params.rng_seed,
        "params_json": json.dumps(asdict(params)),
    }


def save_reservoir(reservoir: dict, out_path: str):
    """Serialize reservoir dict to .npz."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        node_positions=reservoir["node_positions"],
        n_anchors=np.int32(reservoir["n_anchors"]),
        edges=reservoir["edges"],
        rest_lengths=reservoir["rest_lengths"],
        stiffnesses=reservoir["stiffnesses"],
        damping_coefficients=reservoir["damping_coefficients"],
        mesh_origin_in_link=reservoir["mesh_origin_in_link"],
        link_name=reservoir["link_name"],
        rng_seed=np.int32(reservoir["rng_seed"]),
        params_json=reservoir["params_json"],
    )


def load_reservoir(npz_path: str) -> dict:
    """Inverse of save_reservoir. Returns a dict with the same keys."""
    with np.load(npz_path, allow_pickle=False) as z:
        return {
            "node_positions": z["node_positions"],
            "n_anchors": int(z["n_anchors"]),
            "edges": z["edges"],
            "rest_lengths": z["rest_lengths"],
            "stiffnesses": z["stiffnesses"],
            "damping_coefficients": z["damping_coefficients"],
            "mesh_origin_in_link": z["mesh_origin_in_link"],
            "link_name": str(z["link_name"]),
            "rng_seed": int(z["rng_seed"]),
            "params_json": str(z["params_json"]),
        }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_params_from_args(args) -> ReservoirParams:
    base = PRESETS[args.preset]
    # Build a new dataclass, overriding only the fields the user set on the CLI
    fields = asdict(base)
    for name in fields:
        cli_name = name.replace("_", "-")
        cli_value = getattr(args, name.replace("-", "_"), None)
        if cli_value is not None:
            fields[name] = cli_value
    if args.seed is not None:
        fields["rng_seed"] = args.seed
    return ReservoirParams(**fields)


def main():
    p = argparse.ArgumentParser(description="Mesh → reservoir preprocessing")
    p.add_argument("mesh_path", help="Path to .stl / .obj mesh file")
    p.add_argument("output_npz", help="Path to output .npz file")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="medium")
    p.add_argument("--link-name", default="",
                   help="Name of the link this mesh belongs to (metadata)")
    p.add_argument("--seed", type=int, default=None)
    # Overrides (default None so we know whether the user set them)
    for field_name, default in asdict(ReservoirParams()).items():
        if field_name == "rng_seed":
            continue
        p.add_argument(
            f"--{field_name.replace('_', '-')}",
            dest=field_name,
            type=type(default),
            default=None,
            help=f"(default from preset: {default})",
        )
    args = p.parse_args()

    params = _build_params_from_args(args)
    reservoir = mesh_to_reservoir(
        args.mesh_path, params, link_name=args.link_name
    )
    save_reservoir(reservoir, args.output_npz)
    print(f"\nSaved {args.output_npz}")
    print(f"  nodes: {len(reservoir['node_positions'])} "
          f"({reservoir['n_anchors']} anchors + "
          f"{len(reservoir['node_positions']) - reservoir['n_anchors']} interior)")
    print(f"  springs: {len(reservoir['edges'])}")


if __name__ == "__main__":
    main()
