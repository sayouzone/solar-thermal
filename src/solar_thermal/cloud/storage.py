"""다중 클라우드 스토리지 어댑터.

지원 URI
--------
* `/local/absolute/path`
* `file:///local/absolute/path`
* `gs://bucket/key`
* `s3://bucket/key`
* `https://...`  (공개 HTTP 다운로드)

GCS/S3 는 의존성이 optional 이므로 필요한 시점에만 import.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from loguru import logger


_CACHE_DIR = Path(os.environ.get("SOLAR_THERMAL_CACHE", "/tmp/solar-thermal-cache"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_local(uri: str) -> Path:
    """원격 URI 를 로컬 경로로 변환. 필요하면 캐시 디렉터리에 다운로드."""

    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        return Path(parsed.path if scheme == "file" else uri)

    if scheme == "gs":
        return _download_gcs(parsed.netloc, parsed.path.lstrip("/"))

    if scheme == "s3":
        return _download_s3(parsed.netloc, parsed.path.lstrip("/"))

    if scheme in ("http", "https"):
        return _download_http(uri)

    raise ValueError(f"Unsupported URI scheme: {scheme}")


def save_bytes(
    data: bytes,
    prefix: str,
    filename: str,
    backend: str = "local",
    bucket: Optional[str] = None,
) -> str:
    """바이트 데이터를 backend 에 저장하고 URI 를 반환.

    Returns
    -------
    str : 저장된 경로/URI (local path 또는 gs://, s3://)
    """

    if backend == "local":
        out_dir = _CACHE_DIR / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        out_path.write_bytes(data)
        return str(out_path)

    if backend == "gcs":
        if not bucket:
            raise ValueError("GCS backend requires bucket name")
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        blob = client.bucket(bucket).blob(f"{prefix}/{filename}")
        blob.upload_from_string(data)
        return f"gs://{bucket}/{prefix}/{filename}"

    if backend == "s3":
        if not bucket:
            raise ValueError("S3 backend requires bucket name")
        import boto3

        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=f"{prefix}/{filename}", Body=data)
        return f"s3://{bucket}/{prefix}/{filename}"

    raise ValueError(f"Unsupported backend: {backend}")


# ============================================================================ #
# 내부 헬퍼
# ============================================================================ #


def _cache_path(scheme: str, bucket: str, key: str) -> Path:
    safe = key.replace("/", "_")
    return _CACHE_DIR / f"{scheme}_{bucket}_{safe}"


def _download_gcs(bucket: str, key: str) -> Path:
    path = _cache_path("gs", bucket, key)
    if path.exists():
        return path
    from google.cloud import storage  # type: ignore

    client = storage.Client()
    blob = client.bucket(bucket).blob(key)
    logger.info(f"Downloading gs://{bucket}/{key} -> {path}")
    blob.download_to_filename(str(path))
    return path


def _download_s3(bucket: str, key: str) -> Path:
    path = _cache_path("s3", bucket, key)
    if path.exists():
        return path
    import boto3

    s3 = boto3.client("s3")
    logger.info(f"Downloading s3://{bucket}/{key} -> {path}")
    s3.download_file(bucket, key, str(path))
    return path


def _download_http(uri: str) -> Path:
    # URL 해시를 파일명으로
    import hashlib

    h = hashlib.sha1(uri.encode()).hexdigest()[:16]
    suffix = Path(urlparse(uri).path).suffix or ".bin"
    path = _CACHE_DIR / f"http_{h}{suffix}"
    if path.exists():
        return path
    logger.info(f"Downloading {uri} -> {path}")
    with httpx.stream("GET", uri, follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, dir=str(_CACHE_DIR)) as tmp:
            for chunk in r.iter_bytes():
                tmp.write(chunk)
            tmp_path = Path(tmp.name)
    tmp_path.rename(path)
    return path
