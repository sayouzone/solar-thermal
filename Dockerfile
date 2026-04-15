# ========================================================================== #
# 태양광 패널 결함 탐지 서비스 컨테이너
#   - CUDA GPU 추론을 원할 경우 베이스를 nvidia/cuda 런타임으로 변경
# ========================================================================== #
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SOLAR_THERMAL_CONFIG=/app/configs/default.yaml \
    SOLAR_THERMAL_CACHE=/var/cache/solar-thermal

# OpenCV / ORB / tifffile 를 위한 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --no-cache-dir -e . && \
    mkdir -p "$SOLAR_THERMAL_CACHE"

EXPOSE 8080

CMD ["uvicorn", "solar_thermal.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
