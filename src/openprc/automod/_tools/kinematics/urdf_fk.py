"""
Forward kinematics for animating our trajectory bundle.

Thin wrapper over yourdfpy that adapts two bundle-specific conventions
to the library's API:

  1. Joint-name-indexed qpos: our trajectories store /qpos as a (T, n_joints)
     array with column order defined by the `joint_names` HDF5 attr. yourdfpy's
     update_cfg() accepts a dict keyed by joint name, so we build that dict
     from our joint_names + qpos on every frame.

  2. Floating base: yourdfpy parses URDFs with fixed or joint-rooted bases,
     but the Go1 URDF (and every legged-robot URDF we use) has its root link
     sitting at the world origin with no explicit free joint. Real trajectories
     store base pose separately in /base_pose as [x, y, z, qw, qx, qy, qz].
     We apply this base pose ourselves by left-multiplying every link's
     yourdfpy-returned transform with the world-to-base matrix.

The output is always a dict {link_name: (4, 4) float64 matrix} in world frame.

Key APIs:
    RobotFK(urdf_path, joint_names)  — construct once per trajectory
    fk.compute(qpos, base_pose)      — returns {link: 4x4} for one timestep
    fk.link_names                    — list of links with visual meshes
    fk.mesh_info(link_name)          — (mesh_file_path, mesh_origin_4x4, scale)

Why we return link-name-keyed dicts instead of a flat array: downstream code
(the PiViz adapter) needs to know which mesh file to load for each transform.
Keying by link name avoids a parallel list that could drift out of sync.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LinkVisual:
    """One renderable visual belonging to a link."""
    link_name: str
    mesh_path: str          # absolute path to .obj/.stl file
    mesh_origin: np.ndarray  # (4, 4) transform of the mesh within its link frame
    scale: np.ndarray        # (3,) mesh scale; defaults to [1,1,1] if URDF omits it


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert [w, x, y, z] quaternion to a 3x3 rotation matrix.
    Assumes the quaternion is unit-norm (writer validates this).
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def base_pose_to_matrix(base_pose: np.ndarray) -> np.ndarray:
    """
    Turn a (7,) base pose [x, y, z, qw, qx, qy, qz] into a 4x4 world-from-base transform.
    """
    if base_pose.shape != (7,):
        raise ValueError(f"base_pose must be shape (7,), got {base_pose.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rotmat(base_pose[3:7])
    T[:3, 3] = base_pose[:3]
    return T


class RobotFK:
    """
    Forward kinematics driver backed by yourdfpy.

    Construction cost is moderate (URDF parse, mesh discovery); compute() is
    cheap and safe to call at render-frame rate.
    """

    def __init__(
        self,
        urdf_path: str,
        joint_names: Sequence[str],
        floating_base: bool = True,
    ):
        try:
            import yourdfpy  # local import so tests not needing FK don't require it
        except ImportError as e:
            raise ImportError(
                "yourdfpy is required for FK. Install with: pip install yourdfpy"
            ) from e

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self._urdf_path = urdf_path
        # build_scene_graph=True is the default and what we need for FK.
        # load_meshes=False keeps it cheap; we read mesh paths ourselves.
        self._urdf = yourdfpy.URDF.load(
            urdf_path,
            build_scene_graph=True,
            load_meshes=False,
            force_mesh=False,
            build_collision_scene_graph=False,
        )
        self._joint_names: List[str] = list(joint_names)
        self._floating_base = floating_base
        self._unlisted_urdf_joints: List[str] = []  # populated by validation

        self._validate_joint_names()
        self._visuals = self._collect_visuals()
        if not self._visuals:
            raise ValueError(
                f"No visual meshes found in {urdf_path}. PiViz has nothing to render."
            )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    @property
    def link_names(self) -> List[str]:
        """Names of links that have at least one visual mesh."""
        return sorted({v.link_name for v in self._visuals})

    @property
    def visuals(self) -> List[LinkVisual]:
        """One entry per visual mesh. A link may contribute multiple visuals."""
        return list(self._visuals)

    @property
    def base_link_name(self) -> str:
        return self._urdf.base_link

    def compute(
        self,
        qpos: np.ndarray,
        base_pose: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute world-frame 4x4 transforms for every link with a visual.

        Args:
            qpos: (n_joints,) joint positions, ordered per joint_names.
            base_pose: (7,) xyz+wxyz for floating-base robots. Required iff
                       floating_base=True; must be None otherwise.

        Returns:
            Dict mapping link_name → (4, 4) float64 world transform.
        """
        if qpos.shape != (len(self._joint_names),):
            raise ValueError(
                f"qpos has shape {qpos.shape}, expected "
                f"({len(self._joint_names)},) to match joint_names"
            )

        if self._floating_base:
            if base_pose is None:
                # No recorded base pose (common for LeRobot humanoid datasets).
                # Fall back to identity so joint animations still work — the
                # robot will appear stationary at the origin while its joints
                # articulate. This is physically consistent with the recorded
                # proprioceptive data.
                if not getattr(self, "_warned_missing_base", False):
                    print("  note: floating_base=True but no base_pose supplied; "
                          "placing root at world origin (joint motion only)")
                    self._warned_missing_base = True
                T_world_base = np.eye(4, dtype=np.float64)
            else:
                T_world_base = base_pose_to_matrix(base_pose)
        else:
            if base_pose is not None:
                raise ValueError("floating_base=False but base_pose provided")
            T_world_base = np.eye(4, dtype=np.float64)

        # yourdfpy takes a dict keyed by joint name. Build once per call.
        cfg = {name: float(qpos[i]) for i, name in enumerate(self._joint_names)}
        # Hold any URDF actuated joints not covered by this trajectory at 0.
        # (Trajectories for partial-DoF datasets like Dex3 don't record leg joints.)
        for j in self._unlisted_urdf_joints:
            cfg[j] = 0.0
        self._urdf.update_cfg(cfg)

        # get_transform(link_name) returns transform from base_link to link_name
        out: Dict[str, np.ndarray] = {}
        for link_name in self.link_names:
            T_base_link = np.asarray(
                self._urdf.get_transform(link_name, collision_geometry=False),
                dtype=np.float64,
            )
            out[link_name] = T_world_base @ T_base_link
        return out

    def compute_sequence(
        self,
        qpos_seq: np.ndarray,
        base_pose_seq: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Batched version: returns {link_name: (T, 4, 4)} for a whole trajectory.

        Done as a Python loop over timesteps because yourdfpy's FK is not
        vectorized. For Go1 (12 joints, ~20 links, 1500-frame clip) this takes
        a second or two — acceptable for a one-time precomputation.
        """
        T = qpos_seq.shape[0]
        if base_pose_seq is not None and base_pose_seq.shape[0] != T:
            raise ValueError(
                f"qpos_seq has {T} steps but base_pose_seq has {base_pose_seq.shape[0]}"
            )

        out = {n: np.empty((T, 4, 4), dtype=np.float64) for n in self.link_names}
        for t in range(T):
            bp = base_pose_seq[t] if base_pose_seq is not None else None
            frame = self.compute(qpos_seq[t], bp)
            for name, mat in frame.items():
                out[name][t] = mat
        return out

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _validate_joint_names(self) -> None:
        """
        Cross-check joint_names against the URDF's actuated joints.

        Two rules:
        1. Every joint in joint_names must exist as an actuated joint in the URDF
           (otherwise the trajectory is referencing joints the URDF can't handle).
        2. The URDF may have *more* actuated joints than joint_names lists.
           Those extra joints are held at zero during FK. This supports
           partial-DoF datasets (e.g. Dex3 hand manipulation data covers only
           upper body + hands, while the URDF has legs + waist too).
        """
        actuated = [j.name for j in self._urdf.actuated_joints]
        actuated_set = set(actuated)

        missing_in_urdf = [j for j in self._joint_names if j not in actuated_set]
        if missing_in_urdf:
            raise ValueError(
                f"joint_names references joints not actuated in URDF: "
                f"{missing_in_urdf}"
            )

        # Remember which URDF joints are *not* covered by the trajectory data,
        # so compute() can zero-fill them. Sorted for stable output.
        covered = set(self._joint_names)
        self._unlisted_urdf_joints = sorted(j for j in actuated if j not in covered)
        if self._unlisted_urdf_joints:
            print(f"  note: {len(self._unlisted_urdf_joints)} URDF joints "
                  f"not in joint_names — will be held at 0 during FK "
                  f"(e.g. {self._unlisted_urdf_joints[:3]}"
                  f"{'...' if len(self._unlisted_urdf_joints) > 3 else ''})")

    def _collect_visuals(self) -> List[LinkVisual]:
        """Walk the URDF and record every mesh visual with its local-frame origin."""
        urdf_dir = os.path.dirname(os.path.abspath(self._urdf_path))
        visuals: List[LinkVisual] = []

        for link in self._urdf.robot.links:
            for v in link.visuals:
                geom = v.geometry
                if geom is None or geom.mesh is None:
                    continue  # skip primitive (box/cylinder/sphere) visuals
                mesh_ref = geom.mesh.filename

                # Resolve mesh path. URDF conventions:
                #   - "package://foo/bar.stl" (ROS-style) — strip prefix
                #   - "file:///abs/path.stl"             — strip prefix
                #   - "meshes/bar.stl"                   — relative to URDF dir
                #   - "/abs/path.stl"                    — absolute
                if mesh_ref.startswith("package://"):
                    # Best-effort: strip "package://pkg_name/" and resolve
                    # relative to URDF dir. Our own converter emits clean
                    # relative paths so this branch is mainly for upstream URDFs.
                    tail = mesh_ref[len("package://"):]
                    tail = tail.split("/", 1)[1] if "/" in tail else tail
                    mesh_path = os.path.join(urdf_dir, tail)
                elif mesh_ref.startswith("file://"):
                    mesh_path = mesh_ref[len("file://"):]
                elif os.path.isabs(mesh_ref):
                    mesh_path = mesh_ref
                else:
                    mesh_path = os.path.join(urdf_dir, mesh_ref)
                mesh_path = os.path.normpath(mesh_path)

                if not os.path.exists(mesh_path):
                    # Don't hard-fail: warn and skip. A missing mesh for one
                    # link shouldn't break the whole animation.
                    print(f"  warning: mesh not found for link {link.name!r}: "
                          f"{mesh_path}")
                    continue

                # Visual origin: 4x4 local transform applied to the mesh
                # within the link frame. yourdfpy stores this as a matrix already.
                origin = v.origin if v.origin is not None else np.eye(4)
                origin = np.asarray(origin, dtype=np.float64)
                if origin.shape != (4, 4):
                    raise ValueError(
                        f"visual origin for {link.name!r} has shape "
                        f"{origin.shape}, expected (4, 4)"
                    )

                # Mesh scale (URDF: <mesh filename="..." scale="s s s"/>)
                scale_attr = getattr(geom.mesh, "scale", None)
                if scale_attr is None:
                    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
                else:
                    scale = np.asarray(scale_attr, dtype=np.float64).reshape(-1)
                    if scale.shape == (1,):
                        scale = np.repeat(scale, 3)
                    elif scale.shape != (3,):
                        raise ValueError(
                            f"visual scale for {link.name!r} has shape "
                            f"{scale.shape}, expected (3,) or (1,)"
                        )

                visuals.append(LinkVisual(
                    link_name=link.name,
                    mesh_path=mesh_path,
                    mesh_origin=origin,
                    scale=scale,
                ))

        return visuals


def rotmat_to_euler_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract (rx, ry, rz) such that R = Rz(rz) @ Ry(ry) @ Rx(rx).

    This matches PiViz's rotation convention (`Rz × Ry × Rx` applied on the GPU
    per the PiViz README). We output in radians.

    For R = Rz @ Ry @ Rx, the relevant entries are:
        R[2,0] = -sin(ry)
        R[2,1] =  cos(ry) * sin(rx)
        R[2,2] =  cos(ry) * cos(rx)
        R[0,0] =  cos(ry) * cos(rz)
        R[1,0] =  cos(ry) * sin(rz)

    Singularity: when cos(ry) → 0 (i.e., |ry| → π/2), rx and rz become
    coupled. We fix rx=0 and derive rz from R[0,1] and R[0,2], which stays
    stable and keeps the round-trip exact.
    """
    if R.shape != (3, 3):
        raise ValueError(f"expected (3,3), got {R.shape}")

    # ry from R[2,0] = -sin(ry); clip for numerical safety
    neg_sy = float(np.clip(-R[2, 0], -1.0, 1.0))
    ry = float(np.arcsin(neg_sy))

    # Detect gimbal lock: |sin(ry)| ≈ 1 means cos(ry) ≈ 0
    if abs(abs(neg_sy) - 1.0) < 1e-6:
        # cos(ry) ≈ 0: set rx = 0 and recover rz from the remaining entries.
        # With rx=0: R[0,1] = -sz*cx*... simplifies; use atan2 on recoverable entries.
        rx = 0.0
        rz = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        rx = float(np.arctan2(R[2, 1], R[2, 2]))
        rz = float(np.arctan2(R[1, 0], R[0, 0]))
    return rx, ry, rz


def decompose_pose(T: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Split a 4x4 rigid transform into (position, rotation_euler_zyx).
    For PiViz consumption.
    """
    if T.shape != (4, 4):
        raise ValueError(f"expected (4,4), got {T.shape}")
    pos = T[:3, 3].astype(np.float32)
    euler = rotmat_to_euler_zyx(T[:3, :3])
    return pos, euler
