"""Unit tests for spatial clustering and mask partition of unity."""

import numpy as np
import pytest
from fusion.clustering import (
    extract_pixel_features,
    generate_cluster_masks,
    cluster_source_pair,
)


def test_feature_extraction_properties():
    """Verify feature vectors have shape (H, W, 4) and are normalized in [0, 1]."""
    rng = np.random.RandomState(42)
    img = rng.rand(40, 40).astype(np.float32)

    features = extract_pixel_features(img, window_size=5)

    assert features.shape == (40, 40, 4)
    assert features.min() >= 0.0 - 1e-7
    assert features.max() <= 1.0 + 1e-7


def test_partition_of_unity_kmeans():
    """Verify hard cluster masks partition every pixel exactly once (sum_k M_k = 1)."""
    rng = np.random.RandomState(42)
    img = rng.rand(50, 50).astype(np.float32)
    features = extract_pixel_features(img, window_size=7)

    for k in [1, 2, 3, 4, 5]:
        masks = generate_cluster_masks(features, n_clusters=k, method="kmeans", random_state=42)

        assert masks.shape == (k, 50, 50)
        # Binary check: values must be 0 or 1
        unique_vals = np.unique(masks)
        assert set(unique_vals).issubset({0.0, 1.0}), f"Masks contain non-binary values: {unique_vals}"

        # Exact partition of unity: sum across cluster dimension must equal 1 everywhere
        sum_masks = np.sum(masks, axis=0)
        np.testing.assert_allclose(
            sum_masks,
            np.ones((50, 50), dtype=np.float32),
            atol=1e-6,
            err_msg=f"Partition of unity violated for K={k}",
        )


def test_partition_of_unity_fcm():
    """Verify Fuzzy C-Means produces valid hard masks satisfying partition of unity."""
    rng = np.random.RandomState(99)
    img = rng.rand(32, 32).astype(np.float32)
    features = extract_pixel_features(img, window_size=5)

    masks = generate_cluster_masks(features, n_clusters=3, method="fcm", random_state=99)

    assert masks.shape == (3, 32, 32)
    sum_masks = np.sum(masks, axis=0)
    np.testing.assert_allclose(sum_masks, np.ones((32, 32)), atol=1e-6)


def test_clustering_modes_per_source_vs_joint():
    """Verify both per-source and joint clustering modes produce valid masks."""
    rng = np.random.RandomState(77)
    img1 = rng.rand(36, 36).astype(np.float32)
    img2 = rng.rand(36, 36).astype(np.float32)

    # 1. Per-source mode
    m1_per, m2_per = cluster_source_pair(
        img1, img2, n_clusters=3, mode="per_source", random_state=77
    )
    assert m1_per.shape == (3, 36, 36)
    assert m2_per.shape == (3, 36, 36)
    np.testing.assert_allclose(np.sum(m1_per, axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.sum(m2_per, axis=0), 1.0, atol=1e-6)

    # 2. Joint mode: both masks should be identical and valid
    m1_joint, m2_joint = cluster_source_pair(
        img1, img2, n_clusters=3, mode="joint", random_state=77
    )
    assert np.array_equal(m1_joint, m2_joint), "Joint mode masks should be identical for both sources"
    np.testing.assert_allclose(np.sum(m1_joint, axis=0), 1.0, atol=1e-6)
