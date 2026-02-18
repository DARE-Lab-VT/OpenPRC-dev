"""
openprc.vision.cache
====================

Sidecar file caching for feature maps.

Stores detected FeatureMap objects as HDF5 sidecar files next to
the source video. On subsequent runs with the same video + detector
config, the cached map is loaded directly — skipping re-detection.

Cache file naming: ``<video_name>.fmap.h5``
Example: ``experiment_01.mp4`` → ``experiment_01.mp4.fmap.h5``

Validation uses the video's SHA-256 content hash and the detector
configuration to ensure cache integrity.

Dependencies:
    - h5py
    - numpy
    - json (stdlib)

Location: openprc/vision/cache.py

Example
-------
>>> from openprc.vision.cache import FeatureMapCache
>>> from openprc.vision.source import VideoSource
>>> from openprc.vision.detectors import detect
>>>
>>> src = VideoSource("experiment.mp4")
>>> cache = FeatureMapCache(src)
>>>
>>> # Try loading from cache first
>>> fmap = cache.load(detector_name="sift", detector_config={...})
>>> if fmap is None:
...     fmap = detect(src.read_frame(0), method="sift")
...     cache.save(fmap)
>>>
>>> # Or use the convenience method
>>> fmap = cache.get_or_detect(src.read_frame(0), method="sift")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .detectors import FeatureMap

logger = logging.getLogger(__name__)


# ======================================================================
# Sidecar cache
# ======================================================================


class FeatureMapCache:
    """Sidecar HDF5 cache for FeatureMap objects.

    The cache file sits next to the source video and stores one or
    more feature maps keyed by detector name. Each entry includes
    the video content hash and detector config for validation.

    Parameters
    ----------
    source : VideoSource
        The video source (must be a file, not a live camera).
    cache_dir : Path or str or None
        Override directory for the sidecar file. None places it
        next to the video file.
    """

    CACHE_SUFFIX = ".fmap.h5"

    def __init__(self, source, cache_dir: Optional[str] = None):
        if source.is_live:
            raise RuntimeError("Caching is not supported for live camera sources.")

        self._source = source
        self._video_path = source.path

        if cache_dir is not None:
            dir_path = Path(cache_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            self._cache_path = dir_path / (self._video_path.name + self.CACHE_SUFFIX)
        else:
            self._cache_path = self._video_path.parent / (
                    self._video_path.name + self.CACHE_SUFFIX
            )

    @property
    def cache_path(self) -> Path:
        """Path to the sidecar cache file."""
        return self._cache_path

    @property
    def exists(self) -> bool:
        """Whether the cache file exists on disk."""
        return self._cache_path.exists()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, fmap: FeatureMap) -> Path:
        """Save a FeatureMap to the sidecar cache.

        Stored under a group keyed by detector name. Multiple detectors
        can coexist in the same cache file.

        Parameters
        ----------
        fmap : FeatureMap
            Feature map to cache.

        Returns
        -------
        Path
            Path to the cache file.
        """
        content_hash = self._source.content_hash
        group_key = fmap.detector_name

        with h5py.File(self._cache_path, "a") as f:
            # Store video-level metadata
            f.attrs["video_path"] = str(self._video_path)
            f.attrs["content_hash"] = content_hash

            # Remove existing group for this detector if present
            if group_key in f:
                del f[group_key]

            grp = f.create_group(group_key)

            # Arrays
            grp.create_dataset("keypoints", data=fmap.keypoints)
            grp.create_dataset("descriptors", data=fmap.descriptors)
            grp.create_dataset("sizes", data=fmap.sizes)
            grp.create_dataset("angles", data=fmap.angles)
            grp.create_dataset("responses", data=fmap.responses)
            grp.create_dataset("octaves", data=fmap.octaves)

            # Scalar metadata as attributes
            grp.attrs["detector_name"] = fmap.detector_name
            grp.attrs["detector_config"] = json.dumps(fmap.detector_config)
            grp.attrs["ref_frame_index"] = fmap.ref_frame_index
            grp.attrs["frame_shape"] = list(fmap.frame_shape)
            grp.attrs["content_hash"] = content_hash

        logger.info(
            "Cached %d %s features → %s",
            fmap.n_features,
            fmap.detector_name,
            self._cache_path,
        )
        return self._cache_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
            self,
            detector_name: str = "sift",
            detector_config: Optional[dict] = None,
            validate_hash: bool = True,
    ) -> Optional[FeatureMap]:
        """Load a cached FeatureMap if available and valid.

        Parameters
        ----------
        detector_name : str
            Which detector group to load.
        detector_config : dict or None
            If provided, only return a cache hit if the stored config
            matches exactly. None skips config validation.
        validate_hash : bool
            If True, verify the video content hash matches. Default True.

        Returns
        -------
        FeatureMap or None
            Cached feature map, or None on cache miss / validation failure.
        """
        if not self.exists:
            return None

        try:
            with h5py.File(self._cache_path, "r") as f:
                # Hash validation
                if validate_hash:
                    stored_hash = f.attrs.get("content_hash", "")
                    if stored_hash != self._source.content_hash:
                        logger.warning(
                            "Cache hash mismatch — video may have changed. Invalidating."
                        )
                        return None

                if detector_name not in f:
                    return None

                grp = f[detector_name]

                # Config validation
                if detector_config is not None:
                    stored_config = json.loads(grp.attrs.get("detector_config", "{}"))
                    if stored_config != detector_config:
                        logger.info(
                            "Cache config mismatch for %s — re-detection needed.",
                            detector_name,
                        )
                        return None

                # Reconstruct FeatureMap
                frame_shape = tuple(grp.attrs.get("frame_shape", (0, 0)))

                fmap = FeatureMap(
                    keypoints=grp["keypoints"][:],
                    descriptors=grp["descriptors"][:],
                    sizes=grp["sizes"][:],
                    angles=grp["angles"][:],
                    responses=grp["responses"][:],
                    octaves=grp["octaves"][:],
                    detector_name=str(grp.attrs.get("detector_name", detector_name)),
                    detector_config=json.loads(
                        grp.attrs.get("detector_config", "{}")
                    ),
                    ref_frame_index=int(grp.attrs.get("ref_frame_index", 0)),
                    frame_shape=frame_shape,
                )

                logger.info(
                    "Loaded %d cached %s features from %s",
                    fmap.n_features,
                    detector_name,
                    self._cache_path,
                )
                return fmap

        except (OSError, KeyError) as e:
            logger.warning("Cache read failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Convenience: get or detect
    # ------------------------------------------------------------------

    def get_or_detect(
            self,
            frame: np.ndarray,
            method: str = "sift",
            ref_frame_index: int = 0,
            validate_hash: bool = True,
            **detector_kwargs,
    ) -> FeatureMap:
        """Load from cache or detect and cache in one call.

        Parameters
        ----------
        frame : np.ndarray
            Reference frame for detection (used only on cache miss).
        method : str
            Detector name.
        ref_frame_index : int
            Frame index metadata.
        validate_hash : bool
            Whether to verify video hash.
        **detector_kwargs
            Passed to the detector constructor.

        Returns
        -------
        FeatureMap
        """
        from .detectors import get_detector

        detector = get_detector(method, **detector_kwargs)
        config = detector._get_config()

        # Try cache
        cached = self.load(
            detector_name=method,
            detector_config=config,
            validate_hash=validate_hash,
        )
        if cached is not None:
            return cached

        # Detect
        fmap = detector(frame, ref_frame_index=ref_frame_index)

        # Cache
        self.save(fmap)

        return fmap

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def list_cached(self) -> list:
        """List detector names stored in the cache file.

        Returns
        -------
        list of str
        """
        if not self.exists:
            return []

        with h5py.File(self._cache_path, "r") as f:
            return [k for k in f.keys()]

    def invalidate(self, detector_name: Optional[str] = None) -> None:
        """Remove cached data.

        Parameters
        ----------
        detector_name : str or None
            Specific detector to remove. None deletes the entire cache file.
        """
        if not self.exists:
            return

        if detector_name is None:
            self._cache_path.unlink()
            logger.info("Deleted cache file: %s", self._cache_path)
        else:
            with h5py.File(self._cache_path, "a") as f:
                if detector_name in f:
                    del f[detector_name]
                    logger.info("Removed %s from cache.", detector_name)

    def __repr__(self) -> str:
        status = "exists" if self.exists else "empty"
        cached = self.list_cached() if self.exists else []
        return (
            f"FeatureMapCache({self._cache_path.name}, "
            f"status={status}, "
            f"detectors={cached})"
        )