"""
Tests for the pure-math parts of urdf_fk.py.

These don't require yourdfpy — they exercise the quaternion-to-rotmat and
rotmat-to-Euler functions that operate on numpy arrays. The yourdfpy-backed
RobotFK class needs integration tests (against a real URDF), run separately.

Usage:
    python _tools/kinematics/test_fk_math.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from urdf_fk import (
    _quat_wxyz_to_rotmat,
    base_pose_to_matrix,
    rotmat_to_euler_zyx,
    decompose_pose,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def rot_x(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
def rot_y(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
def rot_z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


# -----------------------------------------------------------------------------
# Quaternion tests
# -----------------------------------------------------------------------------

def test_identity_quaternion():
    R = _quat_wxyz_to_rotmat(np.array([1, 0, 0, 0]))
    assert np.allclose(R, np.eye(3)), R
    print("  PASS: identity quaternion → identity matrix")


def test_quaternion_90_around_z():
    # 90° around z: [cos(45°), 0, 0, sin(45°)] = [√2/2, 0, 0, √2/2]
    s = np.sqrt(2) / 2
    R = _quat_wxyz_to_rotmat(np.array([s, 0, 0, s]))
    expected = rot_z(np.pi / 2)
    assert np.allclose(R, expected, atol=1e-7), f"got\n{R}\nexpected\n{expected}"
    print("  PASS: 90° around Z quaternion matches rot_z(π/2)")


def test_quaternion_x_axis_rotation():
    # 30° around x-axis
    a = np.pi / 6
    R = _quat_wxyz_to_rotmat(np.array([np.cos(a/2), np.sin(a/2), 0, 0]))
    expected = rot_x(a)
    assert np.allclose(R, expected, atol=1e-7)
    print("  PASS: 30° around X quaternion matches rot_x")


# -----------------------------------------------------------------------------
# Base-pose matrix tests
# -----------------------------------------------------------------------------

def test_base_pose_identity():
    pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
    T = base_pose_to_matrix(pose)
    assert np.allclose(T, np.eye(4))
    print("  PASS: identity base pose → identity matrix")


def test_base_pose_translation_only():
    pose = np.array([1.2, -3.4, 5.6, 1, 0, 0, 0], dtype=np.float64)
    T = base_pose_to_matrix(pose)
    expected = np.eye(4)
    expected[:3, 3] = [1.2, -3.4, 5.6]
    assert np.allclose(T, expected)
    print("  PASS: base pose with translation only")


def test_base_pose_rotation_plus_translation():
    # 90° around z, translated to (1, 0, 0)
    s = np.sqrt(2) / 2
    pose = np.array([1.0, 0.0, 0.0, s, 0, 0, s], dtype=np.float64)
    T = base_pose_to_matrix(pose)
    # Verify: T @ [1, 0, 0, 1] should give base at (1, 0, 0) then x-axis pointing y
    assert np.allclose(T[:3, 3], [1, 0, 0])
    # The x-axis basis should now point in y
    assert np.allclose(T[:3, 0], [0, 1, 0], atol=1e-7)
    print("  PASS: base pose with rotation + translation composes correctly")


def test_base_pose_rejects_bad_shape():
    try:
        base_pose_to_matrix(np.zeros(6))
    except ValueError as e:
        assert "shape (7,)" in str(e)
        print("  PASS: base_pose_to_matrix rejects wrong shape")
        return
    raise AssertionError("should have raised")


# -----------------------------------------------------------------------------
# Rotation matrix → Euler tests
# -----------------------------------------------------------------------------

def test_identity_to_euler_zero():
    rx, ry, rz = rotmat_to_euler_zyx(np.eye(3))
    assert abs(rx) < 1e-9 and abs(ry) < 1e-9 and abs(rz) < 1e-9
    print("  PASS: identity matrix → zero Euler angles")


def test_pure_axis_rotations_roundtrip():
    """For each principal-axis rotation, R_ex should round-trip through Euler."""
    for axis, rot_fn in [("x", rot_x), ("y", rot_y), ("z", rot_z)]:
        for angle_deg in [10, 30, -45, 80]:
            a = np.deg2rad(angle_deg)
            R = rot_fn(a)
            rx, ry, rz = rotmat_to_euler_zyx(R)
            # Reconstruct: R = Rz @ Ry @ Rx
            R_reconstructed = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
            assert np.allclose(R, R_reconstructed, atol=1e-6), \
                f"axis {axis} angle {angle_deg}°: got ({rx:.3f}, {ry:.3f}, {rz:.3f})"
    print("  PASS: pure-axis rotations round-trip through Euler")


def test_compound_rotation_roundtrip():
    """Rz(30°) @ Ry(20°) @ Rx(10°) should extract cleanly."""
    R = rot_z(np.deg2rad(30)) @ rot_y(np.deg2rad(20)) @ rot_x(np.deg2rad(10))
    rx, ry, rz = rotmat_to_euler_zyx(R)
    assert np.isclose(rx, np.deg2rad(10), atol=1e-6), f"rx: {np.rad2deg(rx)}"
    assert np.isclose(ry, np.deg2rad(20), atol=1e-6), f"ry: {np.rad2deg(ry)}"
    assert np.isclose(rz, np.deg2rad(30), atol=1e-6), f"rz: {np.rad2deg(rz)}"
    print("  PASS: compound Rz@Ry@Rx rotation extracts original angles")


def test_gimbal_lock_handled():
    """When ry = ±π/2, Euler extraction must be stable (not NaN)."""
    # ry near +π/2
    R = rot_y(np.pi / 2 - 1e-8)
    rx, ry, rz = rotmat_to_euler_zyx(R)
    assert np.isfinite(rx) and np.isfinite(ry) and np.isfinite(rz)
    # At exactly π/2
    R = rot_y(np.pi / 2)
    rx, ry, rz = rotmat_to_euler_zyx(R)
    assert np.isfinite(rx) and np.isfinite(ry) and np.isfinite(rz)
    # Reconstructed should still match the original R (even if the specific
    # rx/rz split differs from what we put in)
    R_reconstructed = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    assert np.allclose(R, R_reconstructed, atol=1e-6)
    print("  PASS: gimbal-lock cases return finite values that round-trip")


def test_random_rotations_roundtrip():
    """Fuzz test: 100 random rotations must all round-trip within tolerance."""
    rng = np.random.default_rng(0)
    fails = 0
    for i in range(100):
        # Generate a random unit quaternion
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        R = _quat_wxyz_to_rotmat(q)
        rx, ry, rz = rotmat_to_euler_zyx(R)
        R_back = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
        if not np.allclose(R, R_back, atol=1e-5):
            fails += 1
    assert fails == 0, f"{fails}/100 random rotations failed round-trip"
    print("  PASS: 100 random rotations round-trip through Euler")


def test_decompose_pose():
    """decompose_pose should split 4x4 into position + Euler."""
    T = np.eye(4)
    T[:3, :3] = rot_z(np.pi / 4)
    T[:3, 3] = [1.5, -2.5, 0.3]
    pos, (rx, ry, rz) = decompose_pose(T)
    assert pos.dtype == np.float32
    assert np.allclose(pos, [1.5, -2.5, 0.3])
    assert abs(rx) < 1e-9 and abs(ry) < 1e-9
    assert np.isclose(rz, np.pi / 4)
    print("  PASS: decompose_pose splits correctly and returns float32 position")


def main():
    print("Running FK math tests...\n")
    tests = [
        test_identity_quaternion,
        test_quaternion_90_around_z,
        test_quaternion_x_axis_rotation,
        test_base_pose_identity,
        test_base_pose_translation_only,
        test_base_pose_rotation_plus_translation,
        test_base_pose_rejects_bad_shape,
        test_identity_to_euler_zero,
        test_pure_axis_rotations_roundtrip,
        test_compound_rotation_roundtrip,
        test_gimbal_lock_handled,
        test_random_rotations_roundtrip,
        test_decompose_pose,
    ]
    passed = 0
    for test in tests:
        print(f"- {test.__name__}")
        try:
            test()
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
