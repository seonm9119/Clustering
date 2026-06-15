import numpy as np

from api.utils import (
    SmallNumber,
    compute_squared_distances,
    create_clustering_payload,
    flatten_image_pixels,
    prepare_cluster_image,
    validate_cluster_count,
)


def run_kmeans(image_bytes, cluster_count, max_iter, attempts, seed, color_mode):
    cluster_image, resolved_color_mode = prepare_cluster_image(image_bytes, color_mode)
    pixel_samples = flatten_image_pixels(cluster_image)
    validate_cluster_count(pixel_samples.shape[0], cluster_count)

    centers, labels = compute_kmeans(pixel_samples, cluster_count, max_iter, attempts, seed)

    return create_clustering_payload(
        "kmeans",
        cluster_image,
        labels,
        centers,
        resolved_color_mode,
        {
            "maxIter": max_iter,
            "attempts": attempts,
            "seed": seed
        })


def compute_kmeans(pixel_samples, cluster_count, max_iter, attempts, seed):
    rng = np.random.default_rng(seed)
    best_centers = None
    best_labels = None
    best_inertia = None

    for _ in range(attempts):
        centers = initialize_centers(pixel_samples, cluster_count, rng)
        labels = assign_pixels_to_centers(pixel_samples, centers)

        for _ in range(max_iter):
            previous_centers = centers.copy()
            centers = update_centers(pixel_samples, labels, previous_centers)
            labels = assign_pixels_to_centers(pixel_samples, centers)

            if has_converged(centers, previous_centers):
                break

        inertia = calculate_inertia(pixel_samples, centers, labels)
        if best_inertia is None or inertia < best_inertia:
            best_centers = centers
            best_labels = labels
            best_inertia = inertia

    return best_centers, best_labels


def initialize_centers(pixel_samples, cluster_count, rng):
    pixel_count = pixel_samples.shape[0]
    centers = np.empty((cluster_count, pixel_samples.shape[1]), dtype=np.float32)
    first_center_index = rng.integers(pixel_count)
    centers[0] = pixel_samples[first_center_index]
    closest_squared_distances = compute_squared_distances(pixel_samples, centers[:1]).flatten()

    for center_index in range(1, cluster_count):
        distance_sum = closest_squared_distances.sum()

        if distance_sum <= SmallNumber:
            next_center_index = rng.integers(pixel_count)
        else:
            center_probabilities = closest_squared_distances / distance_sum
            next_center_index = rng.choice(pixel_count, p=center_probabilities)

        centers[center_index] = pixel_samples[next_center_index]
        new_squared_distances = compute_squared_distances(pixel_samples, centers[center_index:center_index + 1]).flatten()
        closest_squared_distances = np.minimum(closest_squared_distances, new_squared_distances)

    return centers


def assign_pixels_to_centers(pixel_samples, centers):
    squared_distances = compute_squared_distances(pixel_samples, centers)

    return squared_distances.argmin(axis=1)


def update_centers(pixel_samples, labels, previous_centers):
    cluster_count = previous_centers.shape[0]
    centers = previous_centers.copy()
    squared_distances = compute_squared_distances(pixel_samples, previous_centers)
    farthest_pixel_indices = np.argsort(squared_distances.min(axis=1))[::-1]
    fallback_index = 0

    for cluster_index in range(cluster_count):
        cluster_pixels = pixel_samples[labels == cluster_index]

        if cluster_pixels.size == 0:
            centers[cluster_index] = pixel_samples[farthest_pixel_indices[fallback_index]]
            fallback_index += 1
        else:
            centers[cluster_index] = cluster_pixels.mean(axis=0)

    return centers


def calculate_inertia(pixel_samples, centers, labels):
    assigned_centers = centers[labels]
    differences = pixel_samples - assigned_centers

    return float(np.sum(differences * differences))


def has_converged(current_centers, previous_centers):
    center_shift = np.max(np.abs(current_centers - previous_centers))

    return center_shift < 1e-4
