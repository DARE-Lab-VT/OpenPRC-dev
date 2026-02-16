"""
openprc.vision.source
=====================

Video source abstraction for file and live camera inputs.

Provides a unified interface for frame iteration, metadata extraction,
and content hashing (used by the sidecar caching system).

Dependencies:
    - opencv-python (cv2)
    - numpy
    - hashlib (stdlib)

Location: openprc/vision/source.py

Example
-------
>>> from openprc.vision.source import VideoSource
>>> src = VideoSource("experiment_01.mp4")
>>> print(src.fps, src.resolution, src.frame_count)
>>> for frame in src:
...     process(frame)
>>> src.release()
"""

import hashlib
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class VideoSource:
    """Unified video source for files and live cameras.

    Abstracts away the source type so downstream pipeline stages
    (detection, tracking, extraction) remain agnostic.

    Parameters
    ----------
    source : str, Path, or int
        File path to a video, or integer camera index for live capture.
    grayscale : bool, optional
        If True, frames are converted to grayscale on read. Default False.

    Attributes
    ----------
    fps : float
        Frames per second.
    resolution : tuple of (int, int)
        (width, height) in pixels.
    frame_count : int
        Total frames (-1 for live cameras).
    is_live : bool
        True if source is a camera.
    path : Path or None
        File path if source is a file, None for cameras.
    """

    def __init__(self, source: Union[str, Path, int], grayscale: bool = False):
        self._grayscale = grayscale
        self._frame_idx = 0
        self._hash: Optional[str] = None

        if isinstance(source, int):
            # Live camera
            self.is_live = True
            self.path = None
            self._cap = cv2.VideoCapture(source)
        else:
            # Video file
            self.is_live = False
            self.path = Path(source).resolve()
            if not self.path.exists():
                raise FileNotFoundError(f"Video not found: {self.path}")
            self._cap = cv2.VideoCapture(str(self.path))

        if not self._cap.isOpened():
            raise IOError(f"Failed to open video source: {source}")

        # Extract metadata from capture
        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS)
        self.resolution: Tuple[int, int] = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.frame_count: int = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else -1
        )

    # ------------------------------------------------------------------
    # Content hashing (for sidecar cache validation)
    # ------------------------------------------------------------------

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the video file for cache validation.

        Computed lazily on first access and cached. Uses a chunked read
        to handle large files without excessive memory usage.

        Returns
        -------
        str
            Hex digest of the file hash.

        Raises
        ------
        RuntimeError
            If source is a live camera (no file to hash).
        """
        if self.is_live:
            raise RuntimeError("Content hash unavailable for live camera sources.")

        if self._hash is None:
            h = hashlib.sha256()
            with open(self.path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB chunks
                    h.update(chunk)
            self._hash = h.hexdigest()

        return self._hash

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """Read the next frame.

        Returns
        -------
        np.ndarray or None
            Frame as (H, W, 3) BGR uint8 array (or (H, W) if grayscale).
            None if no more frames are available.
        """
        ret, frame = self._cap.read()
        if not ret:
            return None

        self._frame_idx += 1

        if self._grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def read_frame(self, index: int) -> Optional[np.ndarray]:
        """Read a specific frame by index (file sources only).

        Parameters
        ----------
        index : int
            Zero-based frame index.

        Returns
        -------
        np.ndarray or None
            The requested frame, or None on failure.

        Raises
        ------
        RuntimeError
            If source is a live camera.
        """
        if self.is_live:
            raise RuntimeError("Random frame access unavailable for live cameras.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        frame = self.read()
        self._frame_idx = index + 1 if frame is not None else index
        return frame

    def seek(self, index: int) -> None:
        """Seek to a frame index without reading.

        Parameters
        ----------
        index : int
            Zero-based frame index.
        """
        if self.is_live:
            raise RuntimeError("Seek unavailable for live cameras.")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self._frame_idx = index

    @property
    def position(self) -> int:
        """Current frame position (zero-based)."""
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        """Iterate over all frames from the current position."""
        return self

    def __next__(self) -> np.ndarray:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def __len__(self) -> int:
        """Total frame count (raises for live cameras)."""
        if self.is_live:
            raise TypeError("Live camera sources have no defined length.")
        return self.frame_count

    # ------------------------------------------------------------------
    # Context manager & cleanup
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Rewind to the first frame (file sources only)."""
        if not self.is_live:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_idx = 0

    def release(self) -> None:
        """Release the underlying video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __del__(self):
        self.release()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        src = f"camera:{self.path}" if self.is_live else str(self.path.name)
        return (
            f"VideoSource({src}, "
            f"fps={self.fps:.1f}, "
            f"res={self.resolution[0]}x{self.resolution[1]}, "
            f"frames={self.frame_count})"
        )

    # ------------------------------------------------------------------
    # Metadata dict (for HDF5 serialization)
    # ------------------------------------------------------------------

    def metadata(self) -> dict:
        """Return source metadata as a plain dict for HDF5 storage.

        Returns
        -------
        dict
            Keys: source_type, path, fps, width, height, frame_count,
            content_hash (file sources only).
        """
        meta = {
            "source_type": "camera" if self.is_live else "file",
            "path": str(self.path) if self.path else "",
            "fps": self.fps,
            "width": self.resolution[0],
            "height": self.resolution[1],
            "frame_count": self.frame_count,
        }
        if not self.is_live:
            meta["content_hash"] = self.content_hash

        return meta