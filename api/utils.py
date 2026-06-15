import base64

import cv2
import numpy as np


SmallNumber = 1e-8


def decode_image(image_bytes):
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise ValueError("Unsupported or corrupted image file.")

    return bgr_image


def prepare_cluster_image(image_bytes, color_mode):
    bgr_image = decode_image(image_bytes)
    resolved_color_mode = resolve_color_mode(bgr_image, color_mode)

    if resolved_color_mode == "gray":
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY), resolved_color_mode

    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), resolved_color_mode


def resolve_color_mode(bgr_image, color_mode):
    if color_mode != "auto":
        return color_mode

    if is_grayscale_image(bgr_image):
        return "gray"

    return "rgb"


def is_grayscale_image(bgr_image):
    blue_channel = bgr_image[:, :, 0].astype(np.int16)
    green_channel = bgr_image[:, :, 1].astype(np.int16)
    red_channel = bgr_image[:, :, 2].astype(np.int16)
    blue_green_gap = np.max(np.abs(blue_channel - green_channel))
    blue_red_gap = np.max(np.abs(blue_channel - red_channel))

    return blue_green_gap <= 2 and blue_red_gap <= 2


def flatten_image_pixels(cluster_image):
    if cluster_image.ndim == 2:
        return cluster_image.reshape((-1, 1)).astype(np.float32)

    return cluster_image.reshape((-1, cluster_image.shape[2])).astype(np.float32)


def validate_cluster_count(pixel_count, cluster_count):
    if cluster_count > pixel_count:
        raise ValueError("clusterCount cannot be greater than the image pixel count.")


def create_clustering_payload(method, cluster_image, labels, centers, color_mode, parameters):
    labels = labels.astype(np.uint8).reshape(-1)
    height, width = cluster_image.shape[:2]
    channel_count = 1 if color_mode == "gray" else 3

    return {
        "method": method,
        "width": int(width),
        "height": int(height),
        "colorMode": color_mode,
        "channelCount": channel_count,
        "clusterCount": int(centers.shape[0]),
        "parameters": parameters,
        "centers": format_centers(centers),
        "labelOrder": order_labels_by_center_brightness(centers, color_mode),
        "labelMap": {
            "dtype": "uint8",
            "encoding": "base64",
            "shape": [int(height), int(width)],
            "data": base64.b64encode(labels.tobytes()).decode("ascii")
        }
    }


def format_centers(centers):
    rounded_centers = np.round(centers.astype(float), 4)

    return rounded_centers.tolist()


def order_labels_by_center_brightness(centers, color_mode):
    if color_mode == "gray":
        brightness = centers[:, 0]
    else:
        brightness = centers @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    return brightness.argsort().astype(int).tolist()


def initialize_membership(pixel_count, cluster_count, rng):
    membership = rng.random((pixel_count, cluster_count), dtype=np.float32)
    membership_sum = membership.sum(axis=1, keepdims=True)
    membership_sum = np.maximum(membership_sum, SmallNumber)

    return membership / membership_sum


def compute_squared_distances(pixel_samples, centers):
    differences = pixel_samples[:, None, :] - centers[None, :, :]
    squared_distances = np.sum(differences * differences, axis=2)

    return np.maximum(squared_distances, SmallNumber)


def compute_weighted_centers(pixel_samples, weights, fuzziness):
    weighted_membership = weights ** fuzziness
    denominator = weighted_membership.sum(axis=0)
    denominator = np.maximum(denominator, SmallNumber)

    return weighted_membership.T @ pixel_samples / denominator[:, None]
