FROM python:3.12-slim

# Tesseract is the offline OCR fallback for packets with no usable text layer.
# EVALUATION.md forbids LLMs/VLMs and cloud OCR; a local engine is explicitly allowed.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
      libtesseract-dev \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY mib/ /app/mib/
COPY policy/ /app/policy/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# The scoring harness mounts the root filesystem read-only; keep all scratch in
# /tmp and never assume the working directory is writable.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TMPDIR=/tmp \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

ENTRYPOINT ["/app/run.sh"]
