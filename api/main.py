import mimetypes
import os
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query
from fastapi.responses import FileResponse

from api.fcm import run_fuzzy_c_means
from api.kmeans import run_kmeans
from api.pcm import run_possibilistic_c_means


app = FastAPI(
    title="Clustering API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None)

CLUSTERING_DATA_DIR = Path(os.environ.get("CLUSTERING_DATA_DIR", "/app/data")).expanduser().resolve(strict=False)
CLUSTERING_DATA_DISPLAY_PATH = os.environ.get(
    "CLUSTERING_DATA_DISPLAY_PATH",
    "/home/nami/repo/gpt_analysis/project/Fault-Extraction-of-Ceramic-Images/Clustering/data")
CLUSTERING_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
CLUSTERING_METHODS = {"kmeans", "fcm", "pcm"}


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
    cluster_count = parse_int_query(cluster_count, "clusterCount", 2, 11)
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


@app.get("/api/clustering/files")
def read_clustering_data_files():
    files = list_clustering_data_files()

    return {
        "success": True,
        "folderPath": str(CLUSTERING_DATA_DIR),
        "displayPath": CLUSTERING_DATA_DISPLAY_PATH,
        "count": len(files),
        "files": files,
    }


@app.get("/api/clustering/image")
def read_clustering_data_image(relative_path=Query("", alias="relativePath")):
    image_path = resolve_clustering_data_file(relative_path)
    media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    return FileResponse(image_path, media_type=media_type)


@app.post("/api/clustering/{method_id}/server-file")
async def create_server_file_clustering_result(
    method_id: str,
    relative_path=Query("", alias="relativePath"),
    cluster_count=Query("2", alias="clusterCount"),
    max_iter=Query("10", alias="maxIter"),
    attempts=Query("10"),
    fuzziness=Query("2.0"),
    color_mode=Query("auto", alias="colorMode"),
    seed=Query(None)):
    image_path = resolve_clustering_data_file(relative_path)
    image_bytes = image_path.read_bytes()
    clustering_payload = run_clustering_method(
        method_id,
        image_bytes,
        cluster_count,
        max_iter,
        attempts,
        fuzziness,
        color_mode,
        seed)
    clustering_payload["sourceFileName"] = image_path.name
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


def run_clustering_method(method_id, image_bytes, cluster_count, max_iter, attempts, fuzziness, color_mode, seed):
    if method_id not in CLUSTERING_METHODS:
        raise HTTPException(status_code=400, detail="Unsupported clustering method.")

    parsed_cluster_count_max = 11 if method_id == "pcm" else 10
    parsed_cluster_count = parse_int_query(cluster_count, "clusterCount", 2, parsed_cluster_count_max)
    parsed_max_iter = parse_int_query(max_iter, "maxIter", 1, 1000)
    parsed_color_mode = parse_color_mode(color_mode)
    parsed_seed = parse_optional_int_query(seed, "seed", 0)

    try:
        if method_id == "kmeans":
            parsed_attempts = parse_int_query(attempts, "attempts", 1, 20)
            return run_kmeans(image_bytes, parsed_cluster_count, parsed_max_iter, parsed_attempts, parsed_seed, parsed_color_mode)

        parsed_fuzziness = parse_float_query(fuzziness, "fuzziness", 1.01, 10.0)

        if method_id == "fcm":
            return run_fuzzy_c_means(image_bytes, parsed_cluster_count, parsed_max_iter, parsed_fuzziness, parsed_seed, parsed_color_mode)

        return run_possibilistic_c_means(image_bytes, parsed_cluster_count, parsed_max_iter, parsed_fuzziness, parsed_seed, parsed_color_mode)
    except ValueError as exception:
        raise HTTPException(status_code=400, detail=str(exception)) from exception


def list_clustering_data_files():
    if not CLUSTERING_DATA_DIR.exists() or not CLUSTERING_DATA_DIR.is_dir():
        raise HTTPException(status_code=400, detail="Clustering data folder not found.")

    files = []

    for current_root, folder_names, file_names in os.walk(CLUSTERING_DATA_DIR):
        folder_names.sort(key=str.lower)

        for file_name in sorted(file_names, key=str.lower):
            file_path = Path(current_root) / file_name

            if file_path.suffix.lower() not in CLUSTERING_IMAGE_EXTENSIONS:
                continue

            relative_path = file_path.relative_to(CLUSTERING_DATA_DIR).as_posix()
            query_string = urllib.parse.urlencode({"relativePath": relative_path})
            files.append({
                "name": file_path.name,
                "relativePath": relative_path,
                "size": file_path.stat().st_size,
                "url": f"/api/clustering/image?{query_string}",
            })

    return files


def resolve_clustering_data_file(relative_path):
    if not CLUSTERING_DATA_DIR.exists() or not CLUSTERING_DATA_DIR.is_dir():
        raise HTTPException(status_code=400, detail="Clustering data folder not found.")

    requested_relative_path = Path(str(relative_path or ""))

    if not str(relative_path or "").strip():
        raise HTTPException(status_code=400, detail="Image file name is required.")

    if requested_relative_path.is_absolute() or ".." in requested_relative_path.parts:
        raise HTTPException(status_code=400, detail="Invalid image file path.")

    image_path = (CLUSTERING_DATA_DIR / requested_relative_path).resolve(strict=False)

    try:
        image_path.relative_to(CLUSTERING_DATA_DIR)
    except ValueError as exception:
        raise HTTPException(status_code=400, detail="Invalid image file path.") from exception

    if image_path.suffix.lower() not in CLUSTERING_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found.")

    return image_path


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
