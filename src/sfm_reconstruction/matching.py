from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import Stage1Dataset
from .geometry import estimate_relative_pose


@dataclass(frozen=True)
class MatchingConfig:
    max_features: int = 1500
    feature_mode: str = "sift"
    sift_contrast_threshold: float = 0.03
    sift_edge_threshold: float = 10.0
    pair_window: int = 3
    ratio_threshold: float = 0.75
    essential_threshold: float = 1.0
    min_inliers: int = 15
    mask_apriltags: bool = False
    apriltag_padding: float = 0.12


@dataclass(frozen=True)
class FeatureSet:
    points: np.ndarray
    descriptors: np.ndarray
    descriptor_type: str
    source: str


@dataclass(frozen=True)
class MatchingSummary:
    candidate_pairs: int
    accepted_pairs: int
    correspondences: int


def circular_image_pairs(
    image_ids: list[int], pair_window: int
) -> list[tuple[int, int]]:
    if pair_window < 1:
        raise ValueError("pair_window must be positive")
    if len(image_ids) < 2:
        return []
    pair_window = min(pair_window, len(image_ids) - 1)
    pairs: set[tuple[int, int]] = set()
    for index, first_id in enumerate(image_ids):
        for offset in range(1, pair_window + 1):
            second_id = image_ids[(index + offset) % len(image_ids)]
            pairs.add(tuple(sorted((first_id, second_id))))
    return sorted(pairs)


def root_sift(descriptors: np.ndarray) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float32)
    sums = np.sum(descriptors, axis=1, keepdims=True)
    normalized = descriptors / np.maximum(sums, 1e-12)
    return np.sqrt(normalized)


def match_descriptors(
    first: np.ndarray,
    second: np.ndarray,
    ratio_threshold: float,
    descriptor_type: str = "float",
) -> np.ndarray:
    """Custom mutual nearest-neighbor matching with Lowe's ratio test."""
    if not 0.0 < ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be between 0 and 1")
    if descriptor_type == "binary":
        return _match_binary_descriptors(first, second, ratio_threshold)
    if descriptor_type != "float":
        raise ValueError(f"Unsupported descriptor type: {descriptor_type}")

    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if len(first) == 0 or len(second) < 2:
        return np.empty((0, 2), dtype=np.int32)

    first_norm = np.sum(first * first, axis=1, keepdims=True)
    second_norm = np.sum(second * second, axis=1)
    distances = first_norm + second_norm - 2.0 * (first @ second.T)
    np.maximum(distances, 0.0, out=distances)

    nearest_two = np.argpartition(distances, kth=1, axis=1)[:, :2]
    nearest_distances = np.take_along_axis(distances, nearest_two, axis=1)
    order = np.argsort(nearest_distances, axis=1)
    nearest_two = np.take_along_axis(nearest_two, order, axis=1)
    nearest_distances = np.take_along_axis(nearest_distances, order, axis=1)
    reverse_best = np.argmin(distances, axis=0)

    accepted = (
        nearest_distances[:, 0]
        < (ratio_threshold * ratio_threshold) * nearest_distances[:, 1]
    )
    first_indices = np.flatnonzero(accepted)
    second_indices = nearest_two[accepted, 0]
    mutual = reverse_best[second_indices] == first_indices
    return np.column_stack(
        (first_indices[mutual], second_indices[mutual])
    ).astype(np.int32)


def _match_binary_descriptors(
    first: np.ndarray,
    second: np.ndarray,
    ratio_threshold: float,
) -> np.ndarray:
    first = np.asarray(first, dtype=np.uint8)
    second = np.asarray(second, dtype=np.uint8)
    if len(first) == 0 or len(second) < 2:
        return np.empty((0, 2), dtype=np.int32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    forward = matcher.knnMatch(first, second, k=2)
    reverse = matcher.knnMatch(second, first, k=1)
    reverse_best = {
        matches[0].queryIdx: matches[0].trainIdx
        for matches in reverse
        if matches
    }
    accepted = []
    for matches in forward:
        if len(matches) < 2:
            continue
        nearest, second_nearest = matches
        if nearest.distance >= ratio_threshold * second_nearest.distance:
            continue
        if reverse_best.get(nearest.trainIdx) == nearest.queryIdx:
            accepted.append((nearest.queryIdx, nearest.trainIdx))
    if not accepted:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(accepted, dtype=np.int32)


def _feature_cache_path(cache_dir: Path, image_id: int, method: str) -> Path:
    return cache_dir / f"{image_id}_{method}.npz"


def _feature_methods(feature_mode: str) -> tuple[str, ...]:
    if feature_mode == "sift":
        return ("sift",)
    if feature_mode == "akaze":
        return ("akaze",)
    if feature_mode in {"sift+akaze", "akaze+sift"}:
        return ("sift", "akaze")
    raise ValueError(
        "feature_mode must be one of: sift, akaze, sift+akaze"
    )


def _feature_mask(
    image: np.ndarray,
    mask_apriltags: bool,
    apriltag_padding: float,
) -> np.ndarray | None:
    if not mask_apriltags:
        return None
    if apriltag_padding < 0.0:
        raise ValueError("apriltag_padding must be non-negative")

    mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    detector = cv2.aruco.ArucoDetector(
        dictionary, cv2.aruco.DetectorParameters()
    )
    corners, _, _ = detector.detectMarkers(image)
    for corner in corners:
        polygon = np.asarray(corner, dtype=np.float32).reshape(4, 2)
        center = polygon.mean(axis=0, keepdims=True)
        polygon = center + (polygon - center) * (1.0 + apriltag_padding)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 0)
    return mask


def _load_cached_feature(
    cache_path: Path,
    method: str,
    config: MatchingConfig,
) -> FeatureSet | None:
    if not cache_path.is_file():
        return None
    cached = np.load(cache_path)
    cached_mask = bool(cached["mask_apriltags"]) if (
        "mask_apriltags" in cached
    ) else False
    cached_padding = float(cached["apriltag_padding"]) if (
        "apriltag_padding" in cached
    ) else 0.0
    cached_contrast = float(cached["sift_contrast_threshold"]) if (
        "sift_contrast_threshold" in cached
    ) else 0.03
    cached_edge = float(cached["sift_edge_threshold"]) if (
        "sift_edge_threshold" in cached
    ) else 10.0
    cached_method = str(cached["method"]) if "method" in cached else method
    if (
        cached_method == method
        and int(cached["max_features"]) == config.max_features
        and cached_mask == config.mask_apriltags
        and np.isclose(cached_padding, config.apriltag_padding)
        and np.isclose(cached_contrast, config.sift_contrast_threshold)
        and np.isclose(cached_edge, config.sift_edge_threshold)
    ):
        descriptor_type = str(cached["descriptor_type"]) if (
            "descriptor_type" in cached
        ) else "float"
        return FeatureSet(
            cached["points"],
            cached["descriptors"],
            descriptor_type,
            method,
        )
    return None


def _detect_feature_set(
    gray: np.ndarray,
    mask: np.ndarray | None,
    method: str,
    config: MatchingConfig,
) -> FeatureSet:
    if method == "sift":
        detector = cv2.SIFT_create(
            nfeatures=config.max_features,
            nOctaveLayers=4,
            contrastThreshold=config.sift_contrast_threshold,
            edgeThreshold=config.sift_edge_threshold,
            sigma=1.6,
        )
        keypoints, descriptors = detector.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) == 0:
            raise RuntimeError("No SIFT features detected")
        return FeatureSet(
            points=np.asarray(
                [keypoint.pt for keypoint in keypoints], dtype=np.float64
            ),
            descriptors=root_sift(descriptors),
            descriptor_type="float",
            source=method,
        )
    if method == "akaze":
        detector = cv2.AKAZE_create()
        keypoints, descriptors = detector.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) == 0:
            raise RuntimeError("No AKAZE features detected")
        if config.max_features > 0 and len(keypoints) > config.max_features:
            responses = np.asarray(
                [keypoint.response for keypoint in keypoints], dtype=np.float64
            )
            selected = np.argsort(responses)[-config.max_features :]
            keypoints = [keypoints[index] for index in selected]
            descriptors = descriptors[selected]
        return FeatureSet(
            points=np.asarray(
                [keypoint.pt for keypoint in keypoints], dtype=np.float64
            ),
            descriptors=np.asarray(descriptors, dtype=np.uint8),
            descriptor_type="binary",
            source=method,
        )
    raise ValueError(f"Unsupported feature method: {method}")


def _extract_features(
    image_path: Path,
    image_id: int,
    cache_dir: Path,
    config: MatchingConfig,
    overwrite: bool,
) -> tuple[FeatureSet, ...]:
    cached_features = []
    methods = _feature_methods(config.feature_mode)
    for method in methods:
        cache_path = _feature_cache_path(cache_dir, image_id, method)
        if not overwrite:
            cached = _load_cached_feature(cache_path, method, config)
            if cached is not None:
                cached_features.append(cached)
                continue
        break
    else:
        return tuple(cached_features)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = _feature_mask(image, config.mask_apriltags, config.apriltag_padding)
    features = []
    for method in methods:
        cache_path = _feature_cache_path(cache_dir, image_id, method)
        if not overwrite:
            cached = _load_cached_feature(cache_path, method, config)
            if cached is not None:
                features.append(cached)
                continue
        feature = _detect_feature_set(gray, mask, method, config)
        np.savez_compressed(
            cache_path,
            method=np.asarray(method),
            points=feature.points,
            descriptors=feature.descriptors,
            descriptor_type=np.asarray(feature.descriptor_type),
            max_features=np.asarray(config.max_features),
            mask_apriltags=np.asarray(config.mask_apriltags),
            apriltag_padding=np.asarray(config.apriltag_padding),
            sift_contrast_threshold=np.asarray(config.sift_contrast_threshold),
            sift_edge_threshold=np.asarray(config.sift_edge_threshold),
        )
        features.append(feature)
    return tuple(features)


def _match_feature_sets(
    first_features: tuple[FeatureSet, ...],
    second_features: tuple[FeatureSet, ...],
    ratio_threshold: float,
) -> np.ndarray:
    correspondences = []
    for first, second in zip(first_features, second_features):
        if first.source != second.source:
            raise ValueError("Feature sources must match")
        descriptor_matches = match_descriptors(
            first.descriptors,
            second.descriptors,
            ratio_threshold,
            first.descriptor_type,
        )
        if len(descriptor_matches) == 0:
            continue
        correspondences.append(
            np.hstack(
                (
                    first.points[descriptor_matches[:, 0]],
                    second.points[descriptor_matches[:, 1]],
                )
            )
        )
    if not correspondences:
        return np.empty((0, 4), dtype=np.float64)
    return np.vstack(correspondences)


def generate_correspondences(
    dataset: Stage1Dataset,
    cache_root: str | Path,
    config: MatchingConfig | None = None,
    overwrite: bool = False,
) -> tuple[Path, MatchingSummary]:
    config = config or MatchingConfig()
    cache_root = Path(cache_root)
    feature_dir = cache_root / "features"
    correspondence_dir = cache_root / "correspondences"
    feature_dir.mkdir(parents=True, exist_ok=True)
    correspondence_dir.mkdir(parents=True, exist_ok=True)

    summary_path = cache_root / "matching_summary.json"
    prior_config = None
    if summary_path.is_file():
        prior_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        prior_config = prior_payload.get("config")
        if prior_config is not None:
            try:
                prior_config = asdict(MatchingConfig(**prior_config))
            except TypeError:
                prior_config = None
        if not overwrite and prior_config == asdict(config):
            summary = MatchingSummary(**prior_payload["summary"])
            return correspondence_dir, summary
    regenerate = overwrite or prior_config != asdict(config)
    if regenerate:
        for path in correspondence_dir.glob("*.txt"):
            path.unlink()

    features = {
        image_id: _extract_features(
            dataset.image_paths[image_id],
            image_id,
            feature_dir,
            config,
            regenerate,
        )
        for image_id in dataset.image_ids
    }
    pairs = circular_image_pairs(dataset.image_ids, config.pair_window)
    accepted_pairs = 0
    correspondence_count = 0

    for pair_index, (first_id, second_id) in enumerate(pairs, start=1):
        output_path = correspondence_dir / f"{first_id}_{second_id}.txt"
        if output_path.is_file() and not regenerate:
            values = np.loadtxt(output_path, ndmin=2)
            accepted_pairs += 1
            correspondence_count += len(values)
            continue

        raw_correspondences = _match_feature_sets(
            features[first_id],
            features[second_id],
            config.ratio_threshold,
        )
        if len(raw_correspondences) >= max(5, config.min_inliers):
            first_points = raw_correspondences[:, :2]
            second_points = raw_correspondences[:, 2:]
            try:
                relative = estimate_relative_pose(
                    first_points,
                    second_points,
                    dataset.intrinsics,
                    config.essential_threshold,
                )
                inliers = relative.inliers
            except (RuntimeError, ValueError):
                inliers = np.zeros(len(raw_correspondences), dtype=bool)
            if int(inliers.sum()) >= config.min_inliers:
                correspondences = np.hstack(
                    (first_points[inliers], second_points[inliers])
                )
                np.savetxt(output_path, correspondences, fmt="%.8f")
                accepted_pairs += 1
                correspondence_count += len(correspondences)

        if pair_index % 25 == 0 or pair_index == len(pairs):
            print(
                f"Matched {pair_index}/{len(pairs)} image pairs; "
                f"accepted {accepted_pairs}."
            )

    summary = MatchingSummary(
        candidate_pairs=len(pairs),
        accepted_pairs=accepted_pairs,
        correspondences=correspondence_count,
    )
    summary_path.write_text(
        json.dumps(
            {"config": asdict(config), "summary": asdict(summary)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return correspondence_dir, summary
