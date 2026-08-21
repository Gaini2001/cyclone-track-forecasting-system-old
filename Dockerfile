# =========================================================
# Cyclone Track Forecasting — API image
#
# Two fixes over the previous version:
#
# 1. curl is installed. The HEALTHCHECK invoked curl, which python:3.11-slim
#    does not ship, so the check failed on every interval. docker-compose then
#    waited on `condition: service_healthy` forever and the dashboard never
#    started. The check now uses Python's own urllib, so nothing extra is
#    needed at all.
#
# 2. models/ is NOT baked in. A Random Forest trained on this dataset runs to
#    several hundred megabytes; copying it into the image produced a
#    multi-gigabyte artifact and rebuilt it on every code change. Models are
#    mounted at runtime instead (see docker-compose.yml).
# =========================================================

FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools are needed to compile some wheels, and are discarded with this
# stage rather than shipped.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# =========================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY --from=builder /install /usr/local

# Application code only. Data, models and figures are mounted at runtime.
COPY src/ src/
COPY app/ app/
COPY main.py .
COPY streamlit_app.py .

# Mount points; empty in the image.
RUN mkdir -p models reports data

# Run as a non-root user.
RUN useradd --create-home --uid 1000 cyclone \
    && chown -R cyclone:cyclone /app
USER cyclone

# urllib rather than curl: no extra package, and it exercises the same code
# path a Python client would.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]