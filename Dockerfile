# syntax=docker/dockerfile:1.7
# Football AI Predictor - one image for the API, the bot, the scheduler and migrations.
#   docker build -t football-ai .
#   docker build --build-arg EXTRAS="[deep]" -t football-ai:deep .   # + PyTorch model
ARG PYTHON_IMAGE=python:3.13-slim

# --------------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
ARG EXTRAS=""
# optional build secret "pip_cert": CA bundle for networks with a TLS-intercepting proxy
RUN --mount=type=secret,id=pip_cert,required=false \
    if [ -f /run/secrets/pip_cert ]; then export PIP_CERT=/run/secrets/pip_cert; fi; \
    pip install ".${EXTRAS}"

# --------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH \
    APP_ENV=production LOG_FORMAT=json DATA_DIR=/app/var LD_LIBRARY_PATH=/opt/gomp
# libgomp (OpenMP, needed by LightGBM) + PostgreSQL client tools (backups).
# Without apt access, the libgomp shipped inside the scikit-learn wheel is used.
RUN (apt-get update \
     && apt-get install -y --no-install-recommends libgomp1 postgresql-client \
     && rm -rf /var/lib/apt/lists/*) || echo "apt unavailable: using vendored libgomp"
COPY --from=builder /opt/venv /opt/venv
# (the vendored copy has a hashed SONAME, so it is exposed under its standard file
# name through LD_LIBRARY_PATH rather than the ldconfig cache)
RUN mkdir -p /opt/gomp \
    && if ! ldconfig -p | grep -q "libgomp.so.1"; then \
      cp "$(ls /opt/venv/lib/python*/site-packages/scikit_learn.libs/libgomp*.so* | head -n1)" \
         /opt/gomp/libgomp.so.1; \
    fi \
    && python -c "import lightgbm, xgboost, sklearn, football_ai.cli"
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY alembic.ini ./
COPY migrations ./migrations
COPY configs ./configs
COPY scripts ./scripts
RUN mkdir -p /app/var && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["football-ai", "serve", "--host", "0.0.0.0", "--port", "8000"]
