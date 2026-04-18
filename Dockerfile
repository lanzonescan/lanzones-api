FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_NO_CACHE_DIR=1 \
	PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
	&& rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PATH="/opt/venv/bin:$PATH" \
	MODEL_PATH=/app/models/best.pt

RUN apt-get update && apt-get install -y --no-install-recommends \
		libglib2.0-0 libgl1 curl ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

ARG WEIGHTS_URL
ARG GITHUB_TOKEN=""
RUN mkdir -p /app/models && \
	if [ -z "$WEIGHTS_URL" ]; then \
		echo 'ERROR: WEIGHTS_URL build arg is not set. In Dokploy this must be under Build Arguments, not Environment.' >&2; \
		exit 1; \
	fi && \
	echo "Fetching weights from: $(echo $WEIGHTS_URL | sed 's/\?.*//')" && \
	if [ -n "$GITHUB_TOKEN" ]; then \
		curl --fail-with-body -sSL --retry 3 --retry-delay 2 --connect-timeout 15 \
			-H "Authorization: Bearer $GITHUB_TOKEN" \
			-H "Accept: application/octet-stream" \
			"$WEIGHTS_URL" -o /app/models/best.pt; \
	else \
		curl --fail-with-body -sSL --retry 3 --retry-delay 2 --connect-timeout 15 \
			"$WEIGHTS_URL" -o /app/models/best.pt; \
	fi && \
	test -s /app/models/best.pt && \
	echo "Downloaded $(stat -c%s /app/models/best.pt 2>/dev/null || stat -f%z /app/models/best.pt) bytes"

RUN chown -R app:app /app
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
	CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "lanzonesscan.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
