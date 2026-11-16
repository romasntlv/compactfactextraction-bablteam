FROM python:3.11-slim AS cpu
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=8
COPY requirements.lock .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.lock
COPY src ./src
COPY configs ./configs
COPY evaluation ./evaluation
ENV DEVICE=cpu CPU_THREADS=4 MODEL_DIR=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS gpu
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=8
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY src ./src
COPY configs ./configs
COPY evaluation ./evaluation
ENV DEVICE=cuda DTYPE=float16 CPU_THREADS=4 MODEL_DIR=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
