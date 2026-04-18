# Lanzones Scan API

A FastAPI service that detects lanzones crop conditions from an uploaded image using a YOLOv8s model. The initial model targets leaf conditions (`dried-leaf`, `healthy`, `leaf-rust`, `powdery-mildew`) and is designed to extend to insect and fruit detection as future models ship.

---

## Table of contents

1. [Overview](#overview)
2. [Tech stack](#tech-stack)
3. [How the system works](#how-the-system-works)
4. [Local environment setup](#local-environment-setup)
5. [Environment variables](#environment-variables)
6. [API reference](#api-reference)
7. [Training](#training)
8. [Testing](#testing)
9. [GitHub setup](#github-setup)
10. [VPS setup (Dokploy host)](#vps-setup-dokploy-host)
11. [Dokploy deployment](#dokploy-deployment)
12. [Operations and troubleshooting](#operations-and-troubleshooting)

---

## Overview

```
client ──HTTPS──▶ Dokploy (Traefik) ──▶ FastAPI container ──▶ YOLOv8 (CPU)
                                          │
                                          ├── JWT auth (HS256, shared secret)
                                          ├── SlowAPI rate limiter (per-sub + per-IP)
                                          └── /app/models/best.pt (baked into image)
```

- Stateless HTTP service, one YOLO model instance per container.
- Weights are fetched from a GitHub Release during Docker build — the repo never contains model binaries.
- No database. Rate-limit state is held in process memory. For predictable limits, run a single replica; multiple replicas enforce limits independently.

### Project layout

```
model-api/
├── Dockerfile                # multi-stage, CPU-only torch, fetches weights at build
├── .dockerignore
├── pyproject.toml            # runtime + dev deps
├── scripts/run_api.sh        # local dev launcher
├── models/.gitkeep           # kept so the folder exists after clone; best.pt is not tracked
├── src/lanzonesscan/
│   ├── api.py                # FastAPI app, /health and /analyze
│   ├── auth.py               # JWT bearer dependency
│   ├── rate_limit.py         # SlowAPI limiter + handler
│   ├── config.py             # env-driven config
│   ├── inference.py          # LanzonesDetector wrapper around ultralytics YOLO
│   ├── train.py              # YOLOv8 training entrypoint (dev only)
│   └── data_setup.py         # unpacks the Roboflow zip (dev only)
└── tests/                    # pytest suite
```

---

## Tech stack

### Language and runtime

- **Python 3.11+** — minimum interpreter version enforced in `pyproject.toml`.
- **Docker** — multi-stage CPU-only image; weights fetched during build via `curl`.
- **Uvicorn** (`uvicorn[standard]>=0.44.0`) — ASGI server, single worker per container.

### Web framework and I/O

| Package | Version floor | Role |
|---|---|---|
| `fastapi` | `>=0.136.0` | HTTP framework, OpenAPI docs, dependency injection |
| `python-multipart` | `>=0.0.26` | `multipart/form-data` parsing for image uploads |
| `pydantic` | `>=2.13.2` | Request/response models and config validation |
| `pillow` | `>=12.2.0` | Image decoding and annotated-image encoding |

### Machine learning

| Package | Version floor | Role |
|---|---|---|
| `ultralytics` | `>=8.4.38` | YOLOv8 model loader, training CLI, prediction pipeline |
| `torch` (CPU) | (transitive) | Tensor backend; pinned to CPU wheels in the Dockerfile |
| `opencv-python-headless` | (transitive) | Image ops used by ultralytics |

### Auth and rate limiting

| Package | Version floor | Role |
|---|---|---|
| `PyJWT` | `>=2.12.1` | HS256/RS256 JWT verification (`iss`, `aud`, `exp`, `nbf`) |
| `slowapi` | `>=0.1.9` | Per-subject and per-IP rate limits with `Retry-After` headers |

### Dev and test

| Package | Version floor | Role |
|---|---|---|
| `pytest` | `>=9.0.3` | Test runner |
| `httpx` | `>=0.28.1` | ASGI test client for FastAPI |

### Infrastructure

- **Dokploy** — app orchestration on the VPS, handles Docker build/deploy.
- **Traefik** (bundled with Dokploy) — TLS termination, Let's Encrypt, routing.
- **GitHub Releases** — weight artifact storage; the repo itself never holds `.pt` files.

---

## How the system works

### Request flow

1. Client sends `POST /analyze` with an image file, optional `conf` and `annotated` query params, and an `Authorization: Bearer <jwt>` header.
2. **Auth** (`auth.py`) validates the JWT against `JWT_SECRET` using the configured algorithm (default HS256), enforces `exp`, and optionally verifies `iss` and `aud`. The token's `sub` claim is attached to `request.state.subject`.
3. **Rate limiting** (`rate_limit.py`) applies two SlowAPI limits:
   - Per JWT subject: `RATE_LIMIT_PER_SUB` (default `10/minute`)
   - Per client IP: `RATE_LIMIT_PER_IP` (default `30/minute`)
   Exceeding either returns `429` with a `Retry-After` header.
4. **Validation** (`api.py`) checks `Content-Type` against `image/jpeg | image/png | image/webp` and rejects bodies larger than 10 MB.
5. **Inference** (`inference.py`) decodes the image with Pillow, runs `YOLO.predict(imgsz=640, conf=conf)`, and converts results to a list of `{class, confidence, bbox}` dicts.
6. If `annotated=true`, the server draws boxes on the image and returns a base64 PNG data URI.
7. A structured log line is emitted with subject, filename, image size, detection count, and inference duration in milliseconds.

### Key invariants

- **`LanzonesDetector` is not thread-safe.** Each container runs a single uvicorn worker (`--workers 1`). Scale horizontally via Dokploy replicas, not vertically via more workers.
- **Model is loaded once at startup** in the FastAPI lifespan hook, not per request.
- **`JWT_SECRET` is required at startup.** The lifespan hook calls `config.require_jwt_secret()` and the process fails fast if it is missing.
- **Weights path is env-driven.** `MODEL_PATH` env var overrides the default `models/best.pt`. In the Docker image it is set to `/app/models/best.pt`.

---

## Local environment setup

### Prerequisites

- Python 3.11+
- `bun` or `curl` for ad-hoc testing
- Apple Silicon optional (used by `train.py`'s MPS default; inference is CPU)
- ~2 GB disk for the virtualenv (`torch`, `ultralytics`, `opencv-python-headless`)

### Install

```bash
cd model-api
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Obtain weights

Place a trained `best.pt` at `models/best.pt`. You can either:

- Train locally (see [Training](#training)), or
- Download from a GitHub release:
  ```bash
  curl -fL -o models/best.pt \
      'https://github.com/<owner>/<repo>/releases/download/weights-v1/best.pt'
  ```

### Run

```bash
export JWT_SECRET='dev-secret-change-me'
./scripts/run_api.sh
# open http://localhost:8000/docs
```

### Generate a dev JWT

```bash
python - <<'PY'
import jwt, time
print(jwt.encode({'sub': 'dev-user', 'exp': int(time.time()) + 3600}, 'dev-secret-change-me', algorithm='HS256'))
PY
```

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
     -F 'file=@leaf.jpg' \
     'http://localhost:8000/analyze?annotated=true' | jq
```

---

## Environment variables

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `JWT_SECRET` | — | yes | HS256 shared secret. Process refuses to start if unset. |
| `JWT_ALGORITHM` | `HS256` | no | Any algorithm supported by PyJWT. Use `RS256` with a public-key-based secret for asymmetric verification. |
| `JWT_LEEWAY_SECONDS` | `0` | no | Clock skew tolerance for `exp` / `nbf`. |
| `JWT_ISSUER` | unset | no | If set, `iss` claim must match. |
| `JWT_AUDIENCE` | unset | no | If set, `aud` claim must match. |
| `RATE_LIMIT_PER_SUB` | `10/minute` | no | Per-JWT-subject limit. SlowAPI syntax. |
| `RATE_LIMIT_PER_IP` | `30/minute` | no | Per-IP limit. |
| `MODEL_PATH` | `./models/best.pt` | no | Absolute path to YOLO weights. Set to `/app/models/best.pt` in the Docker image. |

---

## API reference

### `GET /health`

Unauthenticated. Returns `{"status": "ok", "model_loaded": true}`. Used by Dokploy's healthcheck.

### `POST /analyze`

Authenticated (Bearer JWT). Rate-limited.

**Query parameters**

- `conf` (float, 0–1, default `0.25`) — confidence threshold.
- `annotated` (bool, default `false`) — if true, response includes a base64 PNG with boxes drawn.

**Form data**

- `file` (required) — `image/jpeg`, `image/png`, or `image/webp`, max 10 MB.

**Response 200**

```json
{
  "filename": "leaf.jpg",
  "image_size": {"width": 1024, "height": 768},
  "detections": [
    {"class": "leaf-rust", "confidence": 0.87, "bbox": [x1, y1, x2, y2]}
  ],
  "annotated_image": "data:image/png;base64,..." // null if annotated=false
}
```

**Error responses**

| Status | `detail` | Cause |
|---|---|---|
| 400 | `Invalid image` | Pillow could not decode the upload. |
| 401 | `Missing bearer token` | No `Authorization` header. |
| 401 | `Invalid authorization scheme` | Header is not `Bearer`. |
| 401 | `Token expired` | `exp` in the past. |
| 401 | `Invalid token` | Bad signature, wrong issuer/audience, malformed. |
| 401 | `Token missing subject` | Decoded payload has no string `sub`. |
| 413 | `File exceeds maximum size` | Body > 10 MB. |
| 415 | `Unsupported content-type: ...` | MIME not in the allowlist. |
| 429 | `Rate limit exceeded: ...` | SlowAPI limiter tripped. Includes `Retry-After` header. |
| 500 | `Inference failed` | Unhandled exception during `YOLO.predict`. |

---

## Training

Local-only. Production deploys fetch prebuilt weights from a GitHub release.

```bash
python -m lanzonesscan.train                    # 50 epochs, MPS by default
python -m lanzonesscan.train --epochs 100 --device cpu
```

The dataset is auto-extracted from `../Lanzones.v1i.yolo26.zip` on first run into `data/`. Output weights are copied to `models/best.pt`. Training artifacts (metrics, plots, checkpoint history) land in `models/run/` and are not needed at runtime.

---

## Testing

```bash
pytest
```

The suite covers auth failures, rate-limit headers, data setup, and mocked inference. It does not require a real `best.pt`; a fixture patches `LanzonesDetector`.

---

## GitHub setup

### 1. Push the repo

Create a GitHub repo and push:

```bash
git remote add origin git@github.com:<owner>/<repo>.git
git push -u origin main
```

Model binaries are gitignored (`*.pt`, `models/` except `.gitkeep`) so nothing sensitive or oversized is committed.

### 2. Create a release with the weights

Every time you want to ship a new model:

1. Train or receive a trained `best.pt`.
2. Go to **Releases → Draft a new release**.
3. Tag name: `weights-v1` (or `weights-YYYY-MM-DD`).
4. Attach `best.pt` as a binary asset.
5. Publish.

Copy the asset's download URL — you'll set it as `WEIGHTS_URL` in Dokploy.

- Public repo: `https://github.com/<owner>/<repo>/releases/download/<tag>/best.pt`
- Private repo: same URL, but the Docker build must supply `GITHUB_TOKEN` (PAT with `repo` scope).

### 3. (Optional) Personal access token for private repos

1. GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens**.
2. Scope: single repo, permission `Contents: read`.
3. Save the token; you'll paste it into Dokploy as a build arg.

---

## VPS setup (Dokploy host)

Any Ubuntu 22.04 / 24.04 LTS VPS with at least **2 vCPU / 4 GB RAM / 20 GB disk** works. The model needs roughly 300 MB resident per replica during inference.

### 1. Install Dokploy

```bash
ssh root@<vps-ip>
curl -sSL https://dokploy.com/install.sh | sh
```

The installer sets up Docker, Traefik, and the Dokploy dashboard on port 3000.

### 2. Point a domain at the VPS

Create an `A` record for e.g. `api.example.com` → your VPS IP. Dokploy provisions Let's Encrypt certificates automatically once the domain resolves.

### 3. Firewall

Allow only what's needed:

```bash
ufw allow 22/tcp      # SSH
ufw allow 80/tcp      # HTTP (Traefik, ACME)
ufw allow 443/tcp     # HTTPS
ufw allow 3000/tcp    # Dokploy dashboard (consider restricting to your IP)
ufw enable
```

### 4. Log in to Dokploy

Visit `http://<vps-ip>:3000`, create the admin account, and connect your GitHub account under **Settings → Git Providers**.

---

## Dokploy deployment

### 1. Create the application

- **Type**: Application
- **Source**: GitHub → select your repo and branch (`main`)
- **Build type**: Dockerfile
- **Dockerfile path**: `./Dockerfile`

### 2. Build args

Under **Build → Build arguments**:

| Arg | Value |
|---|---|
| `WEIGHTS_URL` | GitHub release asset URL for `best.pt` |
| `GITHUB_TOKEN` | PAT (leave empty for public repos) |

The Dockerfile `curl`s the weights during build — the image is self-contained, no runtime volume needed.

### 3. Runtime environment variables

Under **Environment**:

```
JWT_SECRET=<generate with: openssl rand -hex 32>
JWT_ALGORITHM=HS256
RATE_LIMIT_PER_SUB=10/minute
RATE_LIMIT_PER_IP=30/minute
```

Optional:

```
JWT_ISSUER=https://auth.yourdomain.com
JWT_AUDIENCE=lanzonesscan
```

Rate-limit state is per-process. If you run more than one replica, each enforces its own counters — users may exceed the nominal limit by hitting different instances. Keep a single replica unless you're willing to tolerate this.

### 4. Networking

- **Exposed port**: `8000`
- **Domain**: `api.example.com`, HTTPS on, certificate via Let's Encrypt
- **Healthcheck**: `GET /health`, expected status `200`

### 5. Deploy

Click **Deploy**. First build takes 3–5 minutes (torch CPU + ultralytics). Subsequent builds are cached unless `pyproject.toml` or `src/` changes.

### 6. Updating the model

1. Attach a new `best.pt` to a new GitHub release.
2. In Dokploy, update `WEIGHTS_URL` to the new asset URL.
3. **Rebuild** (not just redeploy) — build args only take effect on rebuild.

### 7. Scaling

- **Vertical**: useless — one model per uvicorn worker and the detector isn't thread-safe.
- **Horizontal**: increase replicas in Dokploy. Each replica needs ~300 MB RAM for the model plus runtime overhead. Rate-limit counters are per-replica — each replica will independently allow up to its configured limit.

---

## Operations and troubleshooting

### Logs

Structured single-line logs per request:

```
INFO ... analyze sub=user-123 file=leaf.jpg size=1024x768 detections=2 duration_ms=142.3
```

Stream in Dokploy via **Logs** tab, or on the host:

```bash
docker logs -f <container-id>
```

### Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Container exits immediately with `JWT_SECRET env var is required` | Forgot to set `JWT_SECRET` | Add it under Environment in Dokploy |
| Build fails at `curl ... best.pt` | Bad `WEIGHTS_URL` or private repo without token | Verify URL in a browser; add `GITHUB_TOKEN` for private repos |
| Build fails with `401 Bad credentials` | PAT lacks `Contents: read` on the repo | Regenerate token with correct scope |
| `FileNotFoundError: Model weights not found at ...` | `MODEL_PATH` doesn't match where weights landed | Ensure `MODEL_PATH=/app/models/best.pt` in the image (already set by Dockerfile) |
| Inconsistent rate-limit behavior across requests | Each replica has its own in-memory counter | Run a single replica, or raise limits to account for N replicas |
| Slow first request after deploy | Lazy CUDA init / cold torch import | Ignore — model is already loaded; the first `predict` does some per-class setup |
| High memory on a single replica | Ran with `--workers >1` somewhere | Keep `--workers 1`; use replicas instead |

### Health monitoring

Point any uptime monitor (UptimeRobot, BetterStack) at `https://api.example.com/health`. It returns 200 with `model_loaded: true` only after the lifespan hook has loaded weights.

### Rotating the JWT secret

1. Generate a new secret (`openssl rand -hex 32`).
2. Update `JWT_SECRET` in Dokploy.
3. Redeploy. In-flight tokens signed with the old secret immediately stop working — coordinate with the issuer service.
