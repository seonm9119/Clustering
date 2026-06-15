from fastapi import FastAPI, File, HTTPException, Query

from api.fcm import run_fuzzy_c_means
from api.kmeans import run_kmeans
from api.pcm import run_possibilistic_c_means


app = FastAPI(
    title="Clustering API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None)


@app.post("/api/clustering/kmeans")
async def create_kmeans_result(
    image=File(...),
    cluster_count=Query("2", alias="clusterCount"),
    max_iter=Query("10", alias="maxIter"),
    attempts=Query("10"),
    color_mode=Query("auto", alias="colorMode"),
    seed=Query(None)):
    image_bytes = await read_image_bytes(image)
    cluster_count = parse_int_query(cluster_count, "clusterCount", 2, 10)
    max_iter = parse_int_query(max_iter, "maxIter", 1, 1000)
    attempts = parse_int_query(attempts, "attempts", 1, 20)
    color_mode = parse_color_mode(color_mode)
    seed = parse_optional_int_query(seed, "seed", 0)

    try:
        clustering_payload = run_kmeans(image_bytes, cluster_count, max_iter, attempts, seed, color_mode)
    except ValueError as exception:
        raise HTTPException(status_code=400, detail=str(exception)) from exception

    clustering_payload["sourceFileName"] = get_file_name(image)
    return clustering_payload


@app.post("/api/clustering/fcm")
async def create_fcm_result(
    image=File(...),
    cluster_count=Query("2", alias="clusterCount"),
    max_iter=Query("10", alias="maxIter"),
    fuzziness=Query("2.0"),
    color_mode=Query("auto", alias="colorMode"),
    seed=Query(None)):
    image_bytes = await read_image_bytes(image)
    cluster_count = parse_int_query(cluster_count, "clusterCount", 2, 10)
    max_iter = parse_int_query(max_iter, "maxIter", 1, 1000)
    fuzziness = parse_float_query(fuzziness, "fuzziness", 1.01, 10.0)
    color_mode = parse_color_mode(color_mode)
    seed = parse_optional_int_query(seed, "seed", 0)

    try:
        clustering_payload = run_fuzzy_c_means(image_bytes, cluster_count, max_iter, fuzziness, seed, color_mode)
    except ValueError as exception:
        raise HTTPException(status_code=400, detail=str(exception)) from exception

    clustering_payload["sourceFileName"] = get_file_name(image)
    return clustering_payload


@app.post("/api/clustering/pcm")
async def create_pcm_result(
    image=File(...),
    cluster_count=Query("2", alias="clusterCount"),
    max_iter=Query("10", alias="maxIter"),
    fuzziness=Query("2.0"),
    color_mode=Query("auto", alias="colorMode"),
    seed=Query(None)):
    image_bytes = await read_image_bytes(image)
    cluster_count = parse_int_query(cluster_count, "clusterCount", 2, 10)
    max_iter = parse_int_query(max_iter, "maxIter", 1, 1000)
    fuzziness = parse_float_query(fuzziness, "fuzziness", 1.01, 10.0)
    color_mode = parse_color_mode(color_mode)
    seed = parse_optional_int_query(seed, "seed", 0)

    try:
        clustering_payload = run_possibilistic_c_means(image_bytes, cluster_count, max_iter, fuzziness, seed, color_mode)
    except ValueError as exception:
        raise HTTPException(status_code=400, detail=str(exception)) from exception

    clustering_payload["sourceFileName"] = get_file_name(image)
    return clustering_payload


async def read_image_bytes(image):
    if isinstance(image, bytes):
        image_bytes = image
    else:
        image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    return image_bytes


def get_file_name(image):
    return getattr(image, "filename", None)


def parse_int_query(query_value, query_name, min_value, max_value):
    try:
        parsed_value = int(query_value)
    except (TypeError, ValueError) as exception:
        raise HTTPException(status_code=400, detail=f"{query_name} must be an integer.") from exception

    if parsed_value < min_value or parsed_value > max_value:
        raise HTTPException(
            status_code=400,
            detail=f"{query_name} must be between {min_value} and {max_value}.")

    return parsed_value


def parse_optional_int_query(query_value, query_name, min_value):
    if query_value is None or query_value == "":
        return None

    try:
        parsed_value = int(query_value)
    except (TypeError, ValueError) as exception:
        raise HTTPException(status_code=400, detail=f"{query_name} must be an integer.") from exception

    if parsed_value < min_value:
        raise HTTPException(status_code=400, detail=f"{query_name} must be greater than or equal to {min_value}.")

    return parsed_value


def parse_float_query(query_value, query_name, min_value, max_value):
    try:
        parsed_value = float(query_value)
    except (TypeError, ValueError) as exception:
        raise HTTPException(status_code=400, detail=f"{query_name} must be a number.") from exception

    if parsed_value < min_value or parsed_value > max_value:
        raise HTTPException(
            status_code=400,
            detail=f"{query_name} must be between {min_value} and {max_value}.")

    return parsed_value


def parse_color_mode(query_value):
    color_mode = str(query_value).strip().lower()
    allowed_color_modes = {"auto", "rgb", "gray"}

    if color_mode not in allowed_color_modes:
        raise HTTPException(status_code=400, detail="colorMode must be one of auto, rgb, gray.")

    return color_mode
