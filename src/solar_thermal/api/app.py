"""FastAPI 기반 클라우드 결함 탐지 서비스.

엔드포인트
----------
* GET  /healthz                  : liveness
* POST /inspect/uri              : JSON 으로 RGB/IR URI 전달 (gs://, s3://, http)
* POST /inspect/upload           : multipart 로 RGB/IR 파일 업로드
* GET  /jobs/{job_id}            : 비동기 처리 결과 조회

동시 처리량은 asyncio.Semaphore 로 제한한다.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from ..config import AppConfig, load_config
from ..pipeline import DefectDetectionPipeline
from ..schemas import InspectionReport, InspectionRequest


# ---------------------------------------------------------------------- #
# 간단한 in-memory job store (프로덕션은 Redis/Firestore 권장)
# ---------------------------------------------------------------------- #

_JOBS: dict[str, dict] = {}


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """팩토리 패턴으로 FastAPI 앱을 생성."""

    config_path = config_path or os.environ.get("SOLAR_THERMAL_CONFIG", "configs/default.yaml")
    cfg: AppConfig = load_config(config_path)
    pipeline = DefectDetectionPipeline(cfg)

    sem = asyncio.Semaphore(cfg.api.max_concurrent_jobs)
    app = FastAPI(
        title="Solar Panel Defect Detection",
        description="YOLO + VLM fusion inspection for PV modules (RGB + IR)",
        version="0.1.0",
    )

    # -------------------------------------------------------------- #
    # Health
    # -------------------------------------------------------------- #
    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "vlm_enabled": pipeline.vlm is not None}

    # -------------------------------------------------------------- #
    # Sync inference from URIs
    # -------------------------------------------------------------- #
    @app.post("/inspect/uri", response_model=InspectionReport)
    async def inspect_uri(req: InspectionRequest):
        async with sem:
            return await asyncio.to_thread(pipeline.run, req, True)

    # -------------------------------------------------------------- #
    # Sync inference from multipart upload
    # -------------------------------------------------------------- #
    @app.post("/inspect/upload", response_model=InspectionReport)
    async def inspect_upload(
        rgb: UploadFile = File(...),
        ir: UploadFile = File(...),
        ir_format: str = Form("radiometric_tiff"),
        site_id: Optional[str] = Form(None),
        inspection_id: Optional[str] = Form(None),
    ):
        _validate_upload_size(rgb, cfg.api.max_upload_mb)
        _validate_upload_size(ir, cfg.api.max_upload_mb)

        with tempfile.TemporaryDirectory() as td:
            rgb_path = Path(td) / f"rgb_{rgb.filename}"
            ir_path = Path(td) / f"ir_{ir.filename}"
            rgb_path.write_bytes(await rgb.read())
            ir_path.write_bytes(await ir.read())

            req = InspectionRequest(
                rgb_uri=str(rgb_path),
                ir_uri=str(ir_path),
                ir_format=ir_format,
                site_id=site_id,
                inspection_id=inspection_id,
            )
            async with sem:
                return await asyncio.to_thread(pipeline.run, req, True)

    # -------------------------------------------------------------- #
    # Async job submission
    # -------------------------------------------------------------- #
    @app.post("/jobs")
    async def submit_job(req: InspectionRequest, background: BackgroundTasks):
        job_id = req.inspection_id or str(uuid.uuid4())
        _JOBS[job_id] = {"status": "pending", "report": None, "error": None}

        async def _runner():
            async with sem:
                _JOBS[job_id]["status"] = "running"
                try:
                    report = await asyncio.to_thread(pipeline.run, req, True)
                    _JOBS[job_id]["report"] = report.model_dump()
                    _JOBS[job_id]["status"] = "completed"
                except Exception as e:
                    logger.exception(f"Job {job_id} failed")
                    _JOBS[job_id]["status"] = "failed"
                    _JOBS[job_id]["error"] = str(e)

        background.add_task(_runner)
        return {"job_id": job_id, "status": "pending"}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JSONResponse(content=job)

    return app


# ---------------------------------------------------------------------- #
# 내부 검증
# ---------------------------------------------------------------------- #


def _validate_upload_size(file: UploadFile, max_mb: int) -> None:
    if file.size is None:
        return
    if file.size > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"file {file.filename} exceeds {max_mb} MB limit",
        )
