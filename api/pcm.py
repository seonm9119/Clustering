import numpy as np

from api.fcm import compute_fuzzy_c_means
from api.utils import (
    SmallNumber,
    compute_squared_distances,
    compute_weighted_centers,
    create_clustering_payload,
    flatten_image_pixels,
    prepare_cluster_image,
    validate_cluster_count,
)


def run_possibilistic_c_means(image_bytes, cluster_count, max_iter, fuzziness, seed, color_mode):
    cluster_image, resolved_color_mode = prepare_cluster_image(image_bytes, color_mode)
    pixel_samples = flatten_image_pixels(cluster_image)
    validate_cluster_count(pixel_samples.shape[0], cluster_count)

    centers, fuzzy_membership = compute_fuzzy_c_means(pixel_samples, cluster_count, max_iter, fuzziness, seed)
    cluster_scales = compute_cluster_scales(pixel_samples, centers, fuzzy_membership, fuzziness)
    typicality = update_typicality(pixel_samples, centers, cluster_scales, fuzziness)

    for _ in range(max_iter):
        previous_typicality = typicality.copy()
        centers = compute_weighted_centers(pixel_samples, typicality, fuzziness)
        typicality = update_typicality(pixel_samples, centers, cluster_scales, fuzziness)

        if has_converged(typicality, previous_typicality):
            break

    labels = typicality.argmax(axis=1)

    return create_clustering_payload(
        "pcm",
        cluster_image,
        labels,
        centers,
        resolved_color_mode,
        {
            "maxIter": max_iter,
            "fuzziness": fuzziness,
            "seed": seed
        })


def compute_cluster_scales(pixel_samples, centers, membership, fuzziness):
    squared_distances = compute_squared_distances(pixel_samples, centers)
    weighted_membership = membership ** fuzziness
    numerator = np.sum(weighted_membership * squared_distances, axis=0)
    denominator = weighted_membership.sum(axis=0)
    denominator = np.maximum(denominator, SmallNumber)

    return np.maximum(numerator / denominator, SmallNumber)


def update_typicality(pixel_samples, centers, cluster_scales, fuzziness):
    squared_distances = compute_squared_distances(pixel_samples, centers)
    scale_ratio = squared_distances / cluster_scales[None, :]
    power = 1.0 / (fuzziness - 1.0)

    return 1.0 / (1.0 + scale_ratio ** power)


def has_converged(current_typicality, previous_typicality):
    max_change = np.max(np.abs(current_typicality - previous_typicality))

    return max_change < 1e-5
