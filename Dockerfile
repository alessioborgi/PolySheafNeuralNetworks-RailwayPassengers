FROM python:3.13-slim

RUN apt update && apt install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    swig \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel uv
RUN mkdir /.cache && chmod 777 /.cache

WORKDIR /work/project
COPY pyproject.toml uv.lock README.md ./
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen 
ENV PATH="/opt/venv/bin:$PATH"


