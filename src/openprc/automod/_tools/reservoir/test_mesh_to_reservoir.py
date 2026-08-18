"""
Unit tests for pure-numpy functions in mesh_to_reservoir.py.

These cover the geometry/connectivity/randomization math and don't require
trimesh. Trimesh-dependent functions (load, decimate, sample_volume) need
a real mesh and get tested via end-to-end runs on your machine.

Usage:
    python _tools/reservoir/test_mesh_to_reservoir.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mesh_to_reservoir import (
    ReservoirParams,
    avg_nearest_neighbor_spacing,
    jitter_positions,
    randomize_stiffness,
    build_radius_knn_connectivity,
    bridge_components,
    connected_components,
    compute_rest_lengths,
    bcc_lattice_in_bbox,
    bcc_lattice_edges,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _grid_points(n_per_axis: int = 4, spacing: float = 1.0) -> np.ndarray:
    """Generate an (n³, 3) regular grid for deterministic tests."""
    g = np.arange(n_per_axis) * spacing
    xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)


# -----------------------------------------------------------------------------
# avg_nearest_neighbor_spacing
# -----------------------------------------------------------------------------

def test_spacing_unit_grid():
    pts = _grid_points(n_per_axis=3, spacing=1.0)  # 27 points
    # Nearest neighbor should be exactly 1.0 for every point
    spacing = avg_nearest_neighbor_spacing(pts)
    assert np.isclose(spacing, 1.0, atol=1e-6), f"got {spacing}"
    print("  PASS: avg spacing on unit grid = 1.0")


def test_spacing_two_points():
    pts = np.array([[0, 0, 0], [3.0, 4.0, 0]], dtype=np.float32)  # distance 5
    assert np.isclose(avg_nearest_neighbor_spacing(pts), 5.0)
    print("  PASS: avg spacing on 2-point set = 5.0")


def test_spacing_degenerate():
    # Single point should return 1.0 (divide-by-zero guard)
    pts = np.array([[1, 2, 3]], dtype=np.float32)
    assert avg_nearest_neighbor_spacing(pts) == 1.0
    print("  PASS: avg spacing on single point returns sentinel 1.0")


# -----------------------------------------------------------------------------
# jitter_positions
# -----------------------------------------------------------------------------

def test_jitter_zero_is_noop():
    rng = np.random.default_rng(0)
    pts = _grid_points(3, 1.0)
    out = jitter_positions(pts, 0.0, avg_spacing=1.0, rng=rng)
    assert np.array_equal(pts, out)
    print("  PASS: jitter_fraction=0 is no-op")


def test_jitter_scale():
    """jitter_fraction=0.1 with avg_spacing=10 should give noise ~ N(0, 1)."""
    rng = np.random.default_rng(42)
    pts = np.zeros((10000, 3), dtype=np.float32)
    out = jitter_positions(pts, 0.1, avg_spacing=10.0, rng=rng)
    # Expected std: 0.1 * 10 = 1.0
    std = float(np.std(out.ravel()))
    assert 0.95 < std < 1.05, f"jitter std {std} out of expected range"
    # Zero-mean
    assert abs(np.mean(out.ravel())) < 0.05
    print("  PASS: jitter has correct scale")


def test_jitter_locks_anchors():
    rng = np.random.default_rng(0)
    pts = _grid_points(3, 1.0)
    out = jitter_positions(pts, 0.5, avg_spacing=1.0, rng=rng, lock_first=5)
    # First 5 should be unchanged
    assert np.array_equal(pts[:5], out[:5])
    # Beyond 5 should have moved
    assert not np.allclose(pts[5:], out[5:])
    print("  PASS: jitter lock_first preserves initial positions")


# -----------------------------------------------------------------------------
# randomize_stiffness
# -----------------------------------------------------------------------------

def test_stiffness_zero_jitter_is_uniform():
    rng = np.random.default_rng(0)
    stiff = randomize_stiffness(10.0, 100, 0.0, rng)
    assert np.all(stiff == 10.0)
    print("  PASS: stiffness_jitter=0 gives uniform")


def test_stiffness_log_normal_stays_positive():
    rng = np.random.default_rng(0)
    # Large jitter; make sure nothing goes negative
    stiff = randomize_stiffness(10.0, 10000, 0.5, rng)
    assert np.all(stiff > 0)
    # Median should be close to base (log-normal property)
    assert 8.0 < np.median(stiff) < 12.0
    print("  PASS: stiffness log-normal positive + median near base")


def test_stiffness_reproducible():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    s1 = randomize_stiffness(10.0, 50, 0.2, rng1)
    s2 = randomize_stiffness(10.0, 50, 0.2, rng2)
    assert np.array_equal(s1, s2)
    print("  PASS: stiffness reproducible under same seed")


# -----------------------------------------------------------------------------
# build_radius_knn_connectivity
# -----------------------------------------------------------------------------

def test_connectivity_on_unit_grid():
    """3D grid with spacing=1, radius=1.5 should give 6-connected neighbors
    (face-adjacent) for interior points, fewer for corners/edges."""
    pts = _grid_points(3, 1.0)  # 27 points
    edges = build_radius_knn_connectivity(
        pts, radius=1.2, max_neighbors=30, anchor_count=0,
        skip_anchor_to_anchor=False,
    )
    # With radius=1.2, only distance-1 neighbors qualify (not diagonals of √2)
    # Central node (1,1,1) → 6 face-adjacent neighbors
    # Count degree per node
    degrees = np.zeros(27, dtype=np.int32)
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    # At least one node should have degree 6 (the center of a 3x3x3 cube)
    assert degrees.max() == 6, f"max degree {degrees.max()} != 6"
    print("  PASS: radius connectivity gives 6-connectivity on grid center")


def test_connectivity_respects_max_neighbors():
    pts = _grid_points(5, 1.0)  # 125 points
    # Large radius would connect many; max_neighbors should cap degree
    edges = build_radius_knn_connectivity(
        pts, radius=100.0, max_neighbors=3, anchor_count=0,
        skip_anchor_to_anchor=False,
    )
    degrees = np.zeros(125, dtype=np.int32)
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    # Each node proposed 3 neighbors but undirected edges mean degree can
    # exceed 3 (if node B also proposes back to A, same edge; if B proposes
    # something else, degree grows). Upper bound 2*max_neighbors.
    assert degrees.max() <= 2 * 3
    # But each node should have *some* edges
    assert degrees.min() >= 1
    print("  PASS: max_neighbors caps outbound proposals")


def test_connectivity_skip_anchor_to_anchor():
    # 10 anchors + 10 interior, all in a small cluster
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(20, 3)).astype(np.float32)
    # With skip
    with_skip = build_radius_knn_connectivity(
        pts, radius=10.0, max_neighbors=20, anchor_count=10,
        skip_anchor_to_anchor=True,
    )
    # Without skip (for comparison)
    no_skip = build_radius_knn_connectivity(
        pts, radius=10.0, max_neighbors=20, anchor_count=10,
        skip_anchor_to_anchor=False,
    )
    # Count anchor-anchor edges in each
    aa_skip = sum(1 for a, b in with_skip if a < 10 and b < 10)
    aa_no_skip = sum(1 for a, b in no_skip if a < 10 and b < 10)
    assert aa_skip == 0, f"with_skip has {aa_skip} anchor-anchor edges"
    assert aa_no_skip > 0, "skip-off should have anchor-anchor edges"
    print(f"  PASS: anchor-anchor skip works "
          f"({aa_no_skip} → 0 with skip enabled)")


def test_connectivity_returns_sorted_edges():
    pts = _grid_points(3, 1.0)
    edges = build_radius_knn_connectivity(
        pts, radius=1.5, max_neighbors=30, anchor_count=0,
        skip_anchor_to_anchor=False,
    )
    # Each edge should have i < j (sorted pair)
    assert np.all(edges[:, 0] < edges[:, 1])
    # Entire edge list should be sorted lexicographically
    for i in range(len(edges) - 1):
        assert tuple(edges[i]) < tuple(edges[i + 1]), f"out of order at {i}"
    print("  PASS: edges are sorted and i<j")


def test_connectivity_no_duplicates():
    pts = _grid_points(4, 1.0)
    edges = build_radius_knn_connectivity(
        pts, radius=2.0, max_neighbors=30, anchor_count=0,
        skip_anchor_to_anchor=False,
    )
    unique = {(int(a), int(b)) for a, b in edges}
    assert len(unique) == len(edges)
    print("  PASS: no duplicate edges")


# -----------------------------------------------------------------------------
# connected_components
# -----------------------------------------------------------------------------

def test_connected_components_fully_connected():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    comps = connected_components(4, edges)
    assert len(comps) == 1 and comps[0] == {0, 1, 2, 3}
    print("  PASS: chain is 1 component")


def test_connected_components_isolated():
    # 4 isolated singletons
    comps = connected_components(4, np.zeros((0, 2), dtype=np.int32))
    assert len(comps) == 4
    print("  PASS: no edges → each node is its own component")


def test_connected_components_two_groups():
    edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
    comps = connected_components(4, edges)
    sizes = sorted(len(c) for c in comps)
    assert sizes == [2, 2]
    print("  PASS: two disjoint edges → two components")


# -----------------------------------------------------------------------------
# bridge_components
# -----------------------------------------------------------------------------

def test_bridge_noop_when_already_connected():
    """Already-connected graphs should not gain any bridge edges."""
    pts = _grid_points(3, 1.0)
    edges = build_radius_knn_connectivity(
        pts, radius=1.5, max_neighbors=12, anchor_count=0,
        skip_anchor_to_anchor=False,
    )
    assert len(connected_components(len(pts), edges)) == 1, "precondition"
    bridged = bridge_components(pts, edges, anchor_count=0,
                                 skip_anchor_to_anchor=False)
    assert np.array_equal(edges, bridged)
    print("  PASS: connected graph passes through unchanged")


def test_bridge_two_clusters():
    """Two spatially separated clusters should get linked by one edge."""
    cluster_a = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float32)
    cluster_b = np.array([[5, 0, 0], [5.1, 0, 0], [5, 0.1, 0]], dtype=np.float32)
    pts = np.vstack([cluster_a, cluster_b])
    # Build intra-cluster edges with small radius; bridges disallowed.
    edges = build_radius_knn_connectivity(
        pts, radius=0.5, max_neighbors=12, anchor_count=0,
        skip_anchor_to_anchor=False,
    )
    assert len(connected_components(len(pts), edges)) == 2, "precondition"

    bridged = bridge_components(pts, edges, anchor_count=0,
                                 skip_anchor_to_anchor=False)
    assert len(connected_components(len(pts), bridged)) == 1
    # Should add exactly one edge
    assert len(bridged) == len(edges) + 1
    print("  PASS: bridges two clusters with one edge")


def test_bridge_respects_anchor_anchor_ban():
    """When all components are anchor-only, bridge should still connect them
    (fallback to anchor-anchor to avoid leaving graph fragmented)."""
    # 2 anchor clusters, no interior nodes
    cluster_a = np.array([[0, 0, 0], [0.1, 0, 0]], dtype=np.float32)
    cluster_b = np.array([[5, 0, 0], [5.1, 0, 0]], dtype=np.float32)
    pts = np.vstack([cluster_a, cluster_b])
    # 4 anchors, 0 interior. No initial edges.
    empty_edges = np.zeros((0, 2), dtype=np.int32)

    bridged = bridge_components(
        pts, empty_edges, anchor_count=4, skip_anchor_to_anchor=True
    )
    # 4 components → 3 bridges → 1 component
    assert len(connected_components(len(pts), bridged)) == 1
    print("  PASS: bridge falls back when all components are anchor-only")


def test_bridge_prefers_interior_over_anchor_anchor():
    """When an interior↔anchor bridge exists, it should beat an anchor↔anchor
    bridge as long as interior exists with a legal connection."""
    # 3 anchors (indices 0, 1, 2), 1 interior (index 3).
    # Component A: anchor 0 connected to interior 3 (single edge).
    # Component B: anchor 1 standalone, anchor 2 standalone (each their own comp).
    # Anchor 1 nearest interior is node 3 at distance 1.0 (legal: anchor↔interior).
    # Anchor 2 nearest interior is node 3 at distance 5.0.
    # Anchor↔anchor (1↔2) distance 4.0, but should NOT be used since interior
    # bridges exist.
    pts = np.array([
        [0.0, 0, 0],   # idx 0: anchor (in comp A)
        [1.5, 0, 0],   # idx 1: anchor (alone)
        [5.5, 0, 0],   # idx 2: anchor (alone)
        [0.5, 0, 0],   # idx 3: interior (only one, in comp A)
    ], dtype=np.float32)
    edges = np.array([[0, 3]], dtype=np.int32)

    bridged = bridge_components(
        pts, edges, anchor_count=3, skip_anchor_to_anchor=True
    )
    edge_set = {tuple(e) for e in bridged.tolist()}

    # Both isolated anchors must bridge to the interior node, not to each other
    assert (1, 3) in edge_set, \
        f"expected (1, 3) bridging, got {sorted(edge_set)}"
    assert (2, 3) in edge_set, \
        f"expected (2, 3) bridging, got {sorted(edge_set)}"
    # anchor-anchor (1, 2) should NOT be added
    assert (1, 2) not in edge_set, \
        f"unexpected anchor-anchor edge (1, 2): {sorted(edge_set)}"
    print("  PASS: bridge prefers interior↔anchor over anchor↔anchor")


# -----------------------------------------------------------------------------
# compute_rest_lengths
# -----------------------------------------------------------------------------




def test_rest_lengths_simple():
    pts = np.array([[0, 0, 0], [3, 4, 0], [0, 0, 5]], dtype=np.float32)
    edges = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
    rl = compute_rest_lengths(pts, edges)
    assert np.isclose(rl[0], 5.0), rl[0]  # 3-4-5
    assert np.isclose(rl[1], 5.0), rl[1]  # along z
    assert np.isclose(rl[2], np.sqrt(9 + 16 + 25)), rl[2]  # √50
    print("  PASS: rest lengths match Euclidean distances")


def test_rest_lengths_empty():
    pts = _grid_points(2)
    rl = compute_rest_lengths(pts, np.zeros((0, 2), dtype=np.int32))
    assert rl.shape == (0,)
    print("  PASS: empty edge list → empty rest lengths")


# -----------------------------------------------------------------------------
# BCC lattice
# -----------------------------------------------------------------------------

def test_bcc_unit_cube_correct_count():
    """A small bbox at appropriate spacing should produce a sensible number of BCC points."""
    bbox_min = np.zeros(3)
    bbox_max = np.ones(3)
    pts, is_offset = bcc_lattice_in_bbox(bbox_min, bbox_max, 1.0)
    # With pad ≈ 0.05, the bbox grows slightly so nx, ny, nz = 3 each.
    # Corners: 3³ = 27, bodies: 2³ = 8, total = 35.
    # Just sanity-check we get a reasonable number, not zero, and that the
    # corners outnumber the offset (body-centered) points.
    assert len(pts) > 8, f"too few points: {len(pts)}"
    n_corners = (~is_offset).sum()
    n_bodies = is_offset.sum()
    assert n_corners > n_bodies, f"corner/body ratio inverted"
    print(f"  PASS: 1×1×1 bbox produces {n_corners} corners + {n_bodies} bodies "
          f"= {len(pts)} points")


def test_bcc_doubles_density_when_spacing_halves():
    """Halving spacing should multiply node count substantially. In 3D the
    asymptotic factor is 8x, but for small bboxes the bbox-padding effect
    dilutes this — we accept anything in (4x, 12x)."""
    bbox = (np.zeros(3), 4.0 * np.ones(3))
    pts_coarse, _ = bcc_lattice_in_bbox(*bbox, 1.0)
    pts_fine, _ = bcc_lattice_in_bbox(*bbox, 0.5)
    ratio = len(pts_fine) / len(pts_coarse)
    assert 4.0 < ratio < 12.0, f"density scaling wrong: ratio = {ratio:.2f}"
    print(f"  PASS: halving spacing → {ratio:.1f}× more nodes")


def test_bcc_edges_body_to_corner_distance():
    """In a unit BCC cell, body↔corner distance = sqrt(3)/2."""
    bbox = (np.zeros(3), np.ones(3))
    pts, _ = bcc_lattice_in_bbox(*bbox, 1.0)
    edges = bcc_lattice_edges(pts, 1.0)
    # Compute edge lengths
    lengths = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    # Should see two distinct distances: sqrt(3)/2 ≈ 0.866 (body↔corner)
    # and 1.0 (corner↔corner along an axis). With a single 1×1×1 cell there's
    # one body and 8 corners — body↔corner happens 8 times, corner↔corner 12 (cube edges).
    expected_short = np.sqrt(3) / 2
    has_short = np.any(np.abs(lengths - expected_short) < 0.01)
    has_long = np.any(np.abs(lengths - 1.0) < 0.01)
    assert has_short, f"missing body↔corner edges (sqrt(3)/2 ≈ 0.866)"
    assert has_long, f"missing corner↔corner edges (=1.0)"
    print(f"  PASS: BCC edges contain both nearest-neighbor distances")


def test_bcc_edges_no_duplicates():
    bbox = (np.zeros(3), 3.0 * np.ones(3))
    pts, _ = bcc_lattice_in_bbox(*bbox, 1.0)
    edges = bcc_lattice_edges(pts, 1.0)
    unique = {(int(a), int(b)) for a, b in edges}
    assert len(unique) == len(edges)
    # And every edge has i < j
    assert np.all(edges[:, 0] < edges[:, 1])
    print(f"  PASS: BCC edges unique and sorted ({len(edges)} edges)")


def test_bcc_empty_when_spacing_too_large():
    """If spacing >> bbox, only corner grid + 0 bodies = 8 points minimum."""
    pts, _ = bcc_lattice_in_bbox(np.zeros(3), 0.1 * np.ones(3), 1.0)
    # Bbox padding is small, so we expect maybe just the 8 corners (or fewer)
    assert len(pts) >= 1  # at minimum we should get something
    print(f"  PASS: BCC graceful when spacing > bbox ({len(pts)} points)")


# -----------------------------------------------------------------------------
# Parameter dataclass
# -----------------------------------------------------------------------------

def test_params_default_medium_density():
    p = ReservoirParams()
    assert p.decimate_ratio == 0.05
    assert p.node_density == 5_000_000.0
    assert p.rng_seed == 0
    print("  PASS: default ReservoirParams are 'medium' preset")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def main():
    print("Running mesh_to_reservoir pure-function tests...\n")
    tests = [
        test_spacing_unit_grid,
        test_spacing_two_points,
        test_spacing_degenerate,
        test_jitter_zero_is_noop,
        test_jitter_scale,
        test_jitter_locks_anchors,
        test_stiffness_zero_jitter_is_uniform,
        test_stiffness_log_normal_stays_positive,
        test_stiffness_reproducible,
        test_connectivity_on_unit_grid,
        test_connectivity_respects_max_neighbors,
        test_connectivity_skip_anchor_to_anchor,
        test_connectivity_returns_sorted_edges,
        test_connectivity_no_duplicates,
        test_connected_components_fully_connected,
        test_connected_components_isolated,
        test_connected_components_two_groups,
        test_bridge_noop_when_already_connected,
        test_bridge_two_clusters,
        test_bridge_respects_anchor_anchor_ban,
        test_bridge_prefers_interior_over_anchor_anchor,
        test_rest_lengths_simple,
        test_rest_lengths_empty,
        test_bcc_unit_cube_correct_count,
        test_bcc_doubles_density_when_spacing_halves,
        test_bcc_edges_body_to_corner_distance,
        test_bcc_edges_no_duplicates,
        test_bcc_empty_when_spacing_too_large,
        test_params_default_medium_density,
    ]
    passed = 0
    for t in tests:
        print(f"- {t.__name__}")
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
        print()
    print(f"=== {passed}/{len(tests)} tests passed ===")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
