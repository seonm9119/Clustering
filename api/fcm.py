import numpy as np

from api.utils import (
    SmallNumber,
    compute_squared_distances,
    compute_weighted_centers,
    create_clustering_payload,
    flatten_image_pixels,
    initialize_membership,
    prepare_cluster_image,
    validate_cluster_count,
)


def run_fuzzy_c_means(image_bytes, cluster_count, max_iter, fuzziness, seed, color_mode):
    cluster_image, resolved_color_mode = prepare_cluster_image(image_bytes, color_mode)
    pixel_samples = flatten_image_pixels(cluster_image)
    validate_cluster_count(pixel_samples.shape[0], cluster_count)

    centers, membership = compute_fuzzy_c_means(pixel_samples, cluster_count, max_iter, fuzziness, seed)
    labels = membership.argmax(axis=1)

    return create_clustering_payload(
        "fcm",
        cluster_image,
        labels,
        centers,
        resolved_color_mode,
        {
            "maxIter": max_iter,
            "fuzziness": fuzziness,
            "seed": seed
        })


def compute_fuzzy_c_means(pixel_samples, cluster_count, max_iter, fuzziness, seed):
    rng = np.random.default_rng(seed)
    membership = initialize_membership(pixel_samples.shape[0], cluster_count, rng)

    for _ in range(max_iter):
        previous_membership = membership.copy()
        centers = compute_weighted_centers(pixel_samples, membership, fuzziness)
        membership = update_fuzzy_membership(pixel_samples, centers, fuzziness)

        if has_converged(membership, previous_membership):
            break

    centers = compute_weighted_centers(pixel_samples, membership, fuzziness)

    return centers, membership


def update_fuzzy_membership(pixel_samples, centers, fuzziness):
    squared_distances = compute_squared_distances(pixel_samples, centers)
    inverse_power = 1.0 / (fuzziness - 1.0)
    inverse_distances = squared_distances ** -inverse_power
    inverse_distance_sum = inverse_distances.sum(axis=1, keepdims=True)
    inverse_distance_sum = np.maximum(inverse_distance_sum, SmallNumber)

    return inverse_distances / inverse_distance_sum


def has_converged(current_membership, previous_membership):
    max_change = np.max(np.abs(current_membership - previous_membership))

    return max_change < 1e-5
