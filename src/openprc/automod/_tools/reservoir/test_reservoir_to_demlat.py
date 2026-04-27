"""
Tests for the pure-function core of reservoir_to_demlat.py.

Covers:
  - transform_points (4x4 @ point cloud)
  - compose_anchor_signal (world-frame anchor trajectories)
  - upsample_signal (linear interp from trajectory rate to physics rate)
  - offset_edges (global-index offsetting)

Pure numpy, no DEMLAT or h5py dependency. Run from sandbox or anywhere.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reservoir_to_demlat import (
    transform_points,
    compose_anchor_signal,
    upsample_signal,
    offset_edges,
)


# -----------------------------------------------------------------------------
# transform_points
# -----------------------------------------------------------------------------

def test_transform_identity():
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    out = transform_points(pts, np.eye(4))
    assert np.allclose(pts, out)
    print("  PASS: identity transform is a no-op")


def test_transform_translation_only():
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    T = np.eye(4)
    T[:3, 3] = [10, 20, 30]
    out = transform_points(pts, T)
    assert np.allclose(out, pts + [10, 20, 30])
    print("  PASS: translation shifts every point uniformly")


def test_transform_rotation_z_90():
    pts = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    # Rz(90°) sends [1,0,0] → [0,1,0] and [0,1,0] → [-1,0,0]
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    out = transform_points(pts, T)
    assert np.allclose(out[0], [0, 1, 0]), f"got {out[0]}"
    assert np.allclose(out[1], [-1, 0, 0]), f"got {out[1]}"
    print("  PASS: Rz(90°) rotates points correctly")


def test_transform_combined():
    """Translate then rotate vs the equivalent 4x4 should give the same result."""
    pts = np.array([[1, 0, 0]], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]   # Rz(90°)
    T[:3, 3] = [5, 0, 0]
    # T applied to [1,0,0]: rotate then translate → [0,1,0] + [5,0,0] = [5,1,0]
    out = transform_points(pts, T)
    assert np.allclose(out[0], [5, 1, 0])
    print("  PASS: compound rotation + translation correct")


def test_transform_rejects_bad_shapes():
    try:
        transform_points(np.zeros((5,)), np.eye(4))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 1D point array")
    try:
        transform_points(np.zeros((5, 3)), np.eye(3))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 3x3 transform")
    print("  PASS: shape validation rejects bad inputs")


# -----------------------------------------------------------------------------
# compose_anchor_signal
# -----------------------------------------------------------------------------

def test_compose_anchor_static():
    """If T_world_link is identity for all t and mesh_origin is identity,
    anchor positions in world frame == anchor_local."""
    K = 5
    anchors = np.random.default_rng(0).uniform(-1, 1, (K, 3)).astype(np.float32)
    mesh_origin = np.eye(4, dtype=np.float32)
    T_seq = np.tile(np.eye(4, dtype=np.float32), (10, 1, 1))
    out = compose_anchor_signal(anchors, mesh_origin, T_seq)
    assert out.shape == (10, K, 3)
    for t in range(10):
        assert np.allclose(out[t], anchors), f"frame {t} differs"
    print("  PASS: identity transforms keep anchors at local positions")


def test_compose_anchor_translating_link():
    """Link translates uniformly; anchors should follow exactly."""
    K = 3
    anchors = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=np.float32)
    mesh_origin = np.eye(4, dtype=np.float32)
    T_seq = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    # Link moves +x by 1 unit per timestep
    for t in range(4):
        T_seq[t, 0, 3] = float(t)
    out = compose_anchor_signal(anchors, mesh_origin, T_seq)
    # Anchor 0 should go from [0.1, 0, 0] at t=0 to [3.1, 0, 0] at t=3
    assert np.allclose(out[0, 0], [0.1, 0, 0])
    assert np.allclose(out[3, 0], [3.1, 0, 0])
    # Anchor 1's x stays at 0 + t
    assert np.allclose(out[3, 1], [3.0, 0.1, 0.0])
    print("  PASS: translating link moves all anchors by translation")


def test_compose_anchor_rotating_link():
    """Link rotates 90° around z; anchors that were on +x end up on +y."""
    anchors = np.array([[1.0, 0, 0]], dtype=np.float32)
    mesh_origin = np.eye(4, dtype=np.float32)
    T_seq = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    T_seq[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    out = compose_anchor_signal(anchors, mesh_origin, T_seq)
    assert np.allclose(out[0, 0], [1, 0, 0])
    assert np.allclose(out[1, 0], [0, 1, 0])
    print("  PASS: rotating link rotates anchors")


def test_compose_anchor_uses_mesh_origin():
    """Mesh origin offsets the anchor before world transform."""
    anchors = np.array([[0, 0, 0]], dtype=np.float32)
    mesh_origin = np.eye(4, dtype=np.float32)
    mesh_origin[:3, 3] = [10, 0, 0]   # mesh sits at +10 in link frame
    T_seq = np.tile(np.eye(4, dtype=np.float32), (1, 1, 1))   # link at world origin
    out = compose_anchor_signal(anchors, mesh_origin, T_seq)
    # anchor at mesh-frame origin → at +10 in link frame → at +10 in world
    assert np.allclose(out[0, 0], [10, 0, 0])
    print("  PASS: mesh_origin shifts anchors before world transform")


def test_compose_anchor_rejects_bad_shapes():
    K = 4
    anchors = np.zeros((K, 3))
    mesh_origin = np.eye(4)
    T_seq = np.tile(np.eye(4), (5, 1, 1))
    # bad anchors
    try:
        compose_anchor_signal(np.zeros((K,)), mesh_origin, T_seq)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 1D anchors")
    # bad mesh_origin
    try:
        compose_anchor_signal(anchors, np.eye(3), T_seq)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 3x3 mesh_origin")
    # bad T_seq
    try:
        compose_anchor_signal(anchors, mesh_origin, np.zeros((5, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 2D T_seq")
    print("  PASS: compose_anchor_signal validates shapes")


# -----------------------------------------------------------------------------
# upsample_signal
# -----------------------------------------------------------------------------

def test_upsample_doubles_count_at_half_dt():
    """A 50 Hz, 1 second signal upsampled to 100 Hz should yield ~101 samples."""
    t_in = np.arange(0, 1.001, 0.02)   # 51 samples at 50 Hz
    sig = np.zeros((len(t_in), 3), dtype=np.float32)
    sig[:, 0] = t_in   # linearly increasing x
    out, t_out = upsample_signal(sig, t_in, dt_out=0.01)
    assert 100 <= len(out) <= 101
    # Linear interp of a linear signal should be exact
    assert np.allclose(out[:, 0], t_out, atol=1e-6)
    print("  PASS: upsample 50→100 Hz is exact for linear signals")


def test_upsample_preserves_values_at_keyframes():
    """At input timestamps, upsampled values should match input values exactly."""
    t_in = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    sig = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
    out, t_out = upsample_signal(sig, t_in, dt_out=0.05)
    # Find where t_out matches t_in (within tolerance) and check values
    for i, t_kf in enumerate(t_in):
        match = np.argmin(np.abs(t_out - t_kf))
        assert np.allclose(out[match], sig[i], atol=1e-5), \
            f"keyframe {i} at t={t_kf}: expected {sig[i]}, got {out[match]}"
    print("  PASS: upsample preserves keyframe values")


def test_upsample_dtypes():
    t_in = np.linspace(0, 1, 11)
    sig = np.zeros((11, 3), dtype=np.float32)
    out, t_out = upsample_signal(sig, t_in, 0.01)
    assert out.dtype == np.float32
    assert t_out.dtype == np.float64
    print("  PASS: upsample dtypes correct")


def test_upsample_rejects_too_short():
    t_in = np.array([0.0, 1.0])
    sig = np.zeros((2, 3), dtype=np.float32)
    try:
        upsample_signal(sig, t_in, dt_out=10.0)   # would yield <2 samples
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for too-coarse upsample")
    print("  PASS: upsample rejects degenerate output count")


# -----------------------------------------------------------------------------
# offset_edges
# -----------------------------------------------------------------------------

def test_offset_edges_basic():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    out = offset_edges(edges, 100)
    expected = np.array([[100, 101], [101, 102], [102, 103]], dtype=np.int32)
    assert np.array_equal(out, expected)
    print("  PASS: offset_edges adds offset to all indices")


def test_offset_edges_zero():
    edges = np.array([[5, 9]], dtype=np.int32)
    out = offset_edges(edges, 0)
    assert np.array_equal(out, edges)
    # Should be a copy, not a reference
    assert out is not edges
    print("  PASS: offset_edges with 0 returns a copy")


def test_offset_edges_empty():
    edges = np.zeros((0, 2), dtype=np.int32)
    out = offset_edges(edges, 50)
    assert out.shape == (0, 2)
    print("  PASS: offset_edges handles empty input")


def test_offset_edges_dtype():
    """Should preserve int32 dtype."""
    edges = np.array([[1, 2]], dtype=np.int64)
    out = offset_edges(edges, 5)
    assert out.dtype == np.int32
    print("  PASS: offset_edges normalizes to int32")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def main():
    print("Running reservoir_to_demlat pure-function tests...\n")
    tests = [
        test_transform_identity,
        test_transform_translation_only,
        test_transform_rotation_z_90,
        test_transform_combined,
        test_transform_rejects_bad_shapes,
        test_compose_anchor_static,
        test_compose_anchor_translating_link,
        test_compose_anchor_rotating_link,
        test_compose_anchor_uses_mesh_origin,
        test_compose_anchor_rejects_bad_shapes,
        test_upsample_doubles_count_at_half_dt,
        test_upsample_preserves_values_at_keyframes,
        test_upsample_dtypes,
        test_upsample_rejects_too_short,
        test_offset_edges_basic,
        test_offset_edges_zero,
        test_offset_edges_empty,
        test_offset_edges_dtype,
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
