"""
Structural tests for G1 and Panda transforms.

Pure-numpy tests that validate the joint-remap, gripper expansion, time
construction, and train/test split logic without needing lerobot, tfds, or
any other I/O dependencies.

Usage:
    python _tools/test_g1_panda_transforms.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fetch"))

from g1_transforms import (
    build_joint_remap,
    times_from_frame_indices,
    split_episode_train_test,
    slice_long_episode,
)
from panda_transforms import (
    expand_gripper_position,
    select_subset_by_stride,
)


# -----------------------------------------------------------------------------
# G1 transforms
# -----------------------------------------------------------------------------

def test_joint_remap_identity():
    """When source order == target order, remap should be [0, 1, 2, ...]."""
    names = ["j1", "j2", "j3"]
    remap = build_joint_remap(names, names)
    assert list(remap) == [0, 1, 2]
    print("  PASS: joint remap identity")


def test_joint_remap_reorder():
    """Verify the remap indices pick the right source column for each target slot."""
    source = ["elbow", "shoulder", "wrist"]
    target = ["shoulder", "elbow", "wrist"]
    remap = build_joint_remap(source, target)
    # target[0]=shoulder is at source[1], target[1]=elbow is at source[0]
    assert list(remap) == [1, 0, 2]
    # Apply it and check
    sample_source = np.array([[100, 200, 300]], dtype=np.float32)
    reordered = sample_source[:, remap]
    # reordered[0,0] should now be shoulder=200, [0,1]=elbow=100, [0,2]=wrist=300
    assert np.allclose(reordered[0], [200, 100, 300])
    print("  PASS: joint remap reorders source columns correctly")


def test_joint_remap_subset():
    """When target is a subset of source, remap picks the right subset."""
    source = ["a", "b", "c", "d", "e"]
    target = ["c", "a"]
    remap = build_joint_remap(source, target)
    assert list(remap) == [2, 0]
    print("  PASS: joint remap handles target as subset of source")


def test_joint_remap_raises_on_missing():
    try:
        build_joint_remap(["a", "b"], ["a", "xyz"])
    except ValueError as e:
        assert "xyz" in str(e)
        print("  PASS: joint remap raises for missing target joint")
        return
    raise AssertionError("should have raised")


def test_times_uniform_at_30hz():
    t = times_from_frame_indices(90, 30.0)
    assert t.shape == (90,)
    assert t[0] == 0.0
    assert np.isclose(t[-1], 89.0 / 30.0)
    # Uniform spacing check
    assert np.allclose(np.diff(t), 1.0 / 30.0)
    print("  PASS: times_from_frame_indices at 30 Hz")


def test_times_rejects_invalid():
    try:
        times_from_frame_indices(0, 30.0)
    except ValueError:
        pass
    else:
        raise AssertionError("should have raised on n_frames=0")
    try:
        times_from_frame_indices(10, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("should have raised on fps=0")
    print("  PASS: times_from_frame_indices rejects invalid inputs")


def test_train_test_split_sizes():
    train, test = split_episode_train_test(100, test_fraction=0.15)
    assert len(train) + len(test) == 100
    assert len(test) == 15  # ceil(0.15 * 100) = 15
    # Test set is the tail
    assert test == list(range(85, 100))
    print("  PASS: train/test split sizes and tail-placement")


def test_train_test_split_tiny():
    """2 episodes: 1 train, 1 test."""
    train, test = split_episode_train_test(2, test_fraction=0.5)
    assert len(train) == 1
    assert len(test) == 1
    print("  PASS: train/test split handles minimum 2 episodes")


def test_slice_long_episode_short():
    """An episode barely longer than 2x clip_duration should yield 2 clips."""
    # 60s at 30fps = 1800 frames, clip_duration=30s → 2 clips
    clips = slice_long_episode(1800, 30.0, 30.0)
    assert len(clips) == 2
    assert clips[0] == (0, 900)
    assert clips[1] == (900, 1800)
    print("  PASS: slice_long_episode yields expected 2 clips")


def test_slice_long_episode_not_long_enough():
    """An episode of exactly 30s should NOT be sliced."""
    clips = slice_long_episode(900, 30.0, 30.0)  # exactly 30s
    assert len(clips) == 1
    assert clips[0] == (0, 900)
    print("  PASS: slice_long_episode keeps whole-episode when not long enough")


# -----------------------------------------------------------------------------
# Panda transforms
# -----------------------------------------------------------------------------

def test_expand_gripper_closed():
    g = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    out = expand_gripper_position(g, max_opening_m=0.04)
    assert out.shape == (3, 2)
    assert np.all(out == 0.0)
    print("  PASS: expand_gripper at 0 gives zero per-finger")


def test_expand_gripper_open():
    g = np.array([1.0, 1.0], dtype=np.float32)
    out = expand_gripper_position(g, max_opening_m=0.04)
    assert np.allclose(out, 0.04)
    print("  PASS: expand_gripper at 1 gives max opening per-finger")


def test_expand_gripper_half():
    g = np.array([0.5], dtype=np.float32)
    out = expand_gripper_position(g)
    assert np.isclose(out[0, 0], 0.02)
    print("  PASS: expand_gripper at 0.5 gives half opening")


def test_expand_gripper_accepts_2d():
    """Some DROID variants store gripper as (T, 1); handle both shapes."""
    g = np.array([[0.5], [1.0]], dtype=np.float32)
    out = expand_gripper_position(g)
    assert out.shape == (2, 2)
    assert np.isclose(out[1, 0], 0.04)
    print("  PASS: expand_gripper accepts (T, 1) shape")


def test_expand_gripper_clips_out_of_range():
    """Defensive: clip outside-[0,1] values to valid range."""
    g = np.array([-0.5, 1.5, 0.3], dtype=np.float32)
    out = expand_gripper_position(g, max_opening_m=0.04)
    # Use isclose, not ==, because float32 round-trip through multiplication
    # does not land exactly on 0.04 (the actual value is 0.03999999910593033).
    assert np.isclose(out[0, 0], 0.0)      # was -0.5 → clipped to 0 → 0 * 0.04 = 0
    assert np.isclose(out[1, 0], 0.04)     # was 1.5 → clipped to 1 → 1 * 0.04
    assert np.isclose(out[2, 0], 0.012)    # was 0.3 → 0.3 * 0.04
    print("  PASS: expand_gripper clips values outside [0, 1]")


def test_stride_subset_length():
    """Picking 100 from 76000 should give exactly 100 indices in range."""
    idx = select_subset_by_stride(76000, 100)
    assert len(idx) == 100
    assert idx[0] == 0
    assert idx[-1] < 76000
    # Approximately uniform stride
    diffs = np.diff(idx)
    assert diffs.min() >= 1
    assert diffs.max() - diffs.min() <= 2  # within one frame of uniform
    print("  PASS: stride subset length and uniformity")


def test_stride_subset_no_oversample():
    """When target >= source, should return all indices."""
    idx = select_subset_by_stride(50, 100)
    assert list(idx) == list(range(50))
    print("  PASS: stride subset does not oversample")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def main():
    print("Running G1 + Panda transform tests...\n")
    tests = [
        test_joint_remap_identity,
        test_joint_remap_reorder,
        test_joint_remap_subset,
        test_joint_remap_raises_on_missing,
        test_times_uniform_at_30hz,
        test_times_rejects_invalid,
        test_train_test_split_sizes,
        test_train_test_split_tiny,
        test_slice_long_episode_short,
        test_slice_long_episode_not_long_enough,
        test_expand_gripper_closed,
        test_expand_gripper_open,
        test_expand_gripper_half,
        test_expand_gripper_accepts_2d,
        test_expand_gripper_clips_out_of_range,
        test_stride_subset_length,
        test_stride_subset_no_oversample,
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