"""
Structural tests for the Go1 parser transforms.

These test the pure-function core of fetch_go1.py (joint remap, quaternion
reorder, clip slicer, foot-force normalization) with no dependency on a real
bag file or rosbags. Run these before every real-bag ingest so you know the
math is correct.

Usage:
    python _tools/test_go1_transforms.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fetch"))

from go1_transforms import (
    GO1_URDF_JOINT_NAMES,
    LEGKILO_TO_URDF_IDX,
    FOOT_FORCE_LEGKILO_TO_URDF_IDX,
    remap_legkilo_joints_to_urdf,
    reorder_quaternion_xyzw_to_wxyz,
    remap_foot_forces_to_urdf,
    normalize_foot_force_per_sequence,
    compute_contact_flags,
    slice_into_clips,
)


def test_joint_remap_moves_each_slot_correctly():
    """
    Legkilo SDK order:  FR(0,1,2) FL(3,4,5) RL(6,7,8) RR(9,10,11)
    URDF order:         FR(0,1,2) FL(3,4,5) RR(6,7,8) RL(9,10,11)

    Encode each slot's legkilo index as its value, then verify the remap
    returns the right source index at each URDF slot.
    """
    T = 5
    sdk = np.tile(np.arange(12, dtype=np.float32), (T, 1))
    urdf = remap_legkilo_joints_to_urdf(sdk)
    expected = np.tile(LEGKILO_TO_URDF_IDX.astype(np.float32), (T, 1))
    assert np.array_equal(urdf, expected), \
        f"joint remap wrong: got {urdf[0]}, expected {expected[0]}"
    # Spot checks with explicit semantics:
    # URDF[6:9] is RR, and RR lives in SDK[9:12]
    assert urdf[0, 6] == 9 and urdf[0, 7] == 10 and urdf[0, 8] == 11
    # URDF[9:12] is RL, and RL lives in SDK[6:9]
    assert urdf[0, 9] == 6 and urdf[0, 10] == 7 and urdf[0, 11] == 8
    print("  PASS: joint remap slots match FR,FL,RR,RL URDF order")


def test_joint_remap_handles_20_wide_motor_array():
    """Legkilo's motorState is 20 wide; only first 12 are valid."""
    T = 3
    motor20 = np.arange(T * 20, dtype=np.float32).reshape(T, 20)
    urdf = remap_legkilo_joints_to_urdf(motor20)
    assert urdf.shape == (T, 12)
    # First row expected: [0,1,2, 3,4,5, 9,10,11, 6,7,8]
    expected = motor20[0, LEGKILO_TO_URDF_IDX]
    assert np.array_equal(urdf[0], expected)
    print("  PASS: joint remap handles 20-wide motorState")


def test_joint_remap_rejects_short_array():
    try:
        remap_legkilo_joints_to_urdf(np.zeros((5, 8), dtype=np.float32))
    except ValueError as e:
        assert "at least 12" in str(e)
        print("  PASS: joint remap rejects <12 columns")
        return
    print("  FAIL: accepted short array")
    raise AssertionError("should have raised")


def test_quat_reorder_xyzw_to_wxyz():
    """Input [x=1, y=2, z=3, w=4] should become [w=4, x=1, y=2, z=3]."""
    xyzw = np.array([[1, 2, 3, 4]], dtype=np.float32)
    wxyz = reorder_quaternion_xyzw_to_wxyz(xyzw)
    assert np.array_equal(wxyz, np.array([[4, 1, 2, 3]]))
    # Identity quat sanity check
    identity_xyzw = np.array([[0, 0, 0, 1]], dtype=np.float32)
    identity_wxyz = reorder_quaternion_xyzw_to_wxyz(identity_xyzw)
    assert np.array_equal(identity_wxyz, np.array([[1, 0, 0, 0]]))
    print("  PASS: quaternion xyzw -> wxyz")


def test_foot_force_remap():
    """
    Legkilo foot order: FR, FL, RL, RR  → indices [0, 1, 2, 3]
    URDF order:         FR, FL, RR, RL  → want [0, 1, 3, 2]
    """
    ff = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)  # FR,FL,RL,RR
    out = remap_foot_forces_to_urdf(ff)
    # URDF slot 0 = FR (was 0), slot 1 = FL (was 1),
    # slot 2 = RR (was 3), slot 3 = RL (was 2)
    assert np.array_equal(out, np.array([[10.0, 20.0, 40.0, 30.0]])), \
        f"foot force remap wrong: got {out}, expected FR,FL,RR,RL=10,20,40,30"
    print("  PASS: foot force remap FR,FL,RL,RR -> FR,FL,RR,RL")


def test_foot_force_normalize_clips_negatives():
    """Raw int16 readings can be negative (lift-off artifact). We clip to 0."""
    raw = np.array(
        [[-5.0, 10.0, 20.0, -1.0],
         [0.0, 50.0, 100.0, 0.0]],  # max is 100
        dtype=np.float32,
    )
    normed, scale = normalize_foot_force_per_sequence(raw)
    assert scale == 100.0
    # Negatives should be 0 after clipping
    assert normed[0, 0] == 0.0
    assert normed[0, 3] == 0.0
    assert normed[1, 2] == 1.0  # the max
    assert normed[1, 1] == 0.5
    assert normed.max() <= 1.0
    assert normed.min() >= 0.0
    print("  PASS: foot force normalize clips negatives and scales to [0,1]")


def test_foot_force_normalize_all_zeros():
    """Silent sequence shouldn't divide by zero."""
    raw = np.zeros((10, 4), dtype=np.float32)
    normed, scale = normalize_foot_force_per_sequence(raw)
    assert scale == 1.0
    assert np.all(normed == 0.0)
    print("  PASS: foot force normalize handles all-zero sequence")


def test_contact_flags_threshold():
    normed = np.array([[0.0, 0.05, 0.15, 0.9]], dtype=np.float32)
    flags = compute_contact_flags(normed, threshold=0.1)
    assert flags.dtype == bool
    assert np.array_equal(flags[0], [False, False, True, True])
    print("  PASS: contact flags apply threshold")


def test_slice_30s_at_50hz_yields_14_clips_for_420s():
    """
    Simulate legkilo's 50 Hz /high_state stream for 7 minutes (420s).
    30s clips → 14 clips exactly; no tail because 420 is divisible by 30.
    """
    freq = 50.0
    duration = 420.0
    T = int(freq * duration) + 1
    time = np.linspace(0, duration, T)
    clips = slice_into_clips(time, clip_duration_s=30.0)
    assert len(clips) == 14, f"expected 14 clips, got {len(clips)}"
    # Each clip should have roughly 30 * 50 = 1500 samples
    for i, (start, end) in enumerate(clips):
        n = end - start
        assert 1450 <= n <= 1550, f"clip {i} has {n} samples, expected ~1500"
    print(f"  PASS: 30s slicing yields {len(clips)} clips from 420s sequence")


def test_slice_short_sequence_drops_tail():
    """A 35s sequence should yield one 30s clip and drop the 5s tail."""
    time = np.linspace(0, 35, 35 * 50 + 1)
    clips = slice_into_clips(time, clip_duration_s=30.0)
    assert len(clips) == 1
    start, end = clips[0]
    dt = time[end - 1] - time[start]
    assert 29.9 < dt <= 30.0, f"clip duration {dt} not ~30s"
    print("  PASS: 30s slicing drops short tail")


def test_slice_too_short_returns_empty():
    time = np.linspace(0, 5, 251)  # 5s
    clips = slice_into_clips(time, clip_duration_s=30.0)
    assert clips == []
    print("  PASS: sequence shorter than clip duration yields 0 clips")


def test_joint_names_length_matches_remap_idx():
    assert len(GO1_URDF_JOINT_NAMES) == 12
    assert len(LEGKILO_TO_URDF_IDX) == 12
    assert len(FOOT_FORCE_LEGKILO_TO_URDF_IDX) == 4
    # Every URDF index should be a permutation of 0..11
    assert sorted(LEGKILO_TO_URDF_IDX.tolist()) == list(range(12))
    # Foot force remap is a permutation of 0..3
    assert sorted(FOOT_FORCE_LEGKILO_TO_URDF_IDX.tolist()) == [0, 1, 2, 3]
    print("  PASS: constants have consistent sizes and are valid permutations")


def test_writer_integration_with_contact_units():
    """
    End-to-end: build a plausible Go1 clip through the real transforms,
    hand it to write_trajectory, verify the resulting HDF5 has the expected
    attributes. Requires h5py.
    """
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("  SKIP: h5py not installed (run on your machine)")
        return True

    import tempfile
    sys.path.insert(0, HERE)
    from write_trajectory import write_trajectory

    T = 1500
    time = np.arange(T) / 50.0
    rng = np.random.default_rng(0)
    qpos = 0.01 * rng.standard_normal((T, 12)).astype(np.float32)
    qvel = 0.1 * rng.standard_normal((T, 12)).astype(np.float32)
    tau = 1.0 * rng.standard_normal((T, 12)).astype(np.float32)
    base_pose = np.zeros((T, 7), dtype=np.float32)
    base_pose[:, 3] = 1.0  # identity quat wxyz
    ff_raw = np.maximum(0, rng.integers(-10, 200, size=(T, 4))).astype(np.float32)
    ff_norm, scale = normalize_foot_force_per_sequence(ff_raw)
    flags = compute_contact_flags(ff_norm)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "go1_clip.h5")
        write_trajectory(
            out_path=out,
            robot_name="go1",
            trajectory_id="test_clip",
            joint_names=GO1_URDF_JOINT_NAMES,
            contact_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            time=time,
            qpos=qpos,
            qvel=qvel,
            tau=tau,
            base_pose=base_pose,
            contact_foot_force=ff_norm,
            contact_flags=flags,
            contact_force_units="normalized_per_sequence",
            contact_force_scale=f"divided by {scale:.0f}",
            source="legkilo",
            source_type="real_hardware",
            source_url="https://github.com/ouguangjun/legkilo-dataset",
            source_citation="Ou et al., RA-L 2024",
            source_license="MIT",
        )
        with h5py.File(out, "r") as f:
            assert f.attrs["contact_force_units"] == "normalized_per_sequence"
            assert "divided by" in f.attrs["contact_force_scale"]
            assert "contact/foot_force" in f.attrs["signals_present"]
            assert "contact/contact_flags" in f.attrs["signals_present"]
            # qpos shape
            assert f["qpos"].shape == (T, 12)
            assert f["contact/foot_force"].shape == (T, 4)
    print("  PASS: writer integration with contact units")


def test_writer_rejects_foot_force_without_units():
    """Foot force must come with contact_force_units."""
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("  SKIP: h5py not installed")
        return True

    import tempfile
    sys.path.insert(0, HERE)
    from write_trajectory import write_trajectory

    T = 10
    time = np.arange(T) / 50.0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            write_trajectory(
                out_path=os.path.join(tmp, "bad.h5"),
                robot_name="go1",
                trajectory_id="bad",
                joint_names=GO1_URDF_JOINT_NAMES,
                contact_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                time=time,
                qpos=np.zeros((T, 12), dtype=np.float32),
                contact_foot_force=np.zeros((T, 4), dtype=np.float32),
                # Missing contact_force_units
                source="legkilo",
                source_type="real_hardware",
                source_url="https://example.invalid",
                source_citation="test",
                source_license="MIT",
            )
        except ValueError as e:
            assert "contact_force_units must be provided" in str(e)
            print("  PASS: writer rejects foot force without units")
            return True
    print("  FAIL: writer accepted foot force without units")
    raise AssertionError("should have raised")


def main():
    print("Running Go1 transform tests...\n")
    tests = [
        test_joint_names_length_matches_remap_idx,
        test_joint_remap_moves_each_slot_correctly,
        test_joint_remap_handles_20_wide_motor_array,
        test_joint_remap_rejects_short_array,
        test_quat_reorder_xyzw_to_wxyz,
        test_foot_force_remap,
        test_foot_force_normalize_clips_negatives,
        test_foot_force_normalize_all_zeros,
        test_contact_flags_threshold,
        test_slice_30s_at_50hz_yields_14_clips_for_420s,
        test_slice_short_sequence_drops_tail,
        test_slice_too_short_returns_empty,
        test_writer_integration_with_contact_units,
        test_writer_rejects_foot_force_without_units,
    ]
    passed = 0
    for test in tests:
        print(f"- {test.__name__}")
        try:
            result = test()
            passed += 1 if (result is None or result) else 0
        except Exception as e:
            import traceback
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()
        print()

    total = len(tests)
    print(f"=== {passed}/{total} tests passed ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
