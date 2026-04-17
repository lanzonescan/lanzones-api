# Lanzones Leaf Scan API

Detects lanzones leaf conditions (`dried-leaf`, `healthy`, `leaf-rust`, `powdery-mildew`) from an uploaded image using a YOLOv8s model trained on the Roboflow `lanzones-qdaaq` dataset.

## Setup

```bash
cd model-api
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Train

The dataset is auto-extracted from `../Lanzones.v1i.yolo26.zip` on first run.

```bash
python -m lanzonesscan.train                    # 50 epochs, MPS by default
python -m lanzonesscan.train --epochs 100 --device cpu
```

Output: `models/best.pt`.

## Serve

Requires `JWT_SECRET` in the environment. Startup fails fast if it's missing.

```bash
export JWT_SECRET='your-strong-shared-secret'
# optional overrides:
# export JWT_ALGORITHM=HS256
# export JWT_ISSUER='https://auth.example.com'
# export JWT_AUDIENCE='lanzonesscan'
# export RATE_LIMIT_STORAGE_URI='redis://localhost:6379/0'
# export RATE_LIMIT_PER_SUB='10/minute'
# export RATE_LIMIT_PER_IP='30/minute'

./scripts/run_api.sh                            # http://localhost:8000/docs
```

## Endpoints

- `GET /health` — open, returns `{"status": "ok", "model_loaded": true}`
- `POST /analyze` — **requires** `Authorization: Bearer <jwt>` header signed with `JWT_SECRET` (HS256 by default). Rate-limited to 10 req/min per JWT `sub` and 30 req/min per IP.
  - Query params: `conf` (float, 0-1, default 0.25), `annotated` (bool, default false)

Example:
```bash
TOKEN=$(your-jwt-issuer)
curl -s -H "Authorization: Bearer $TOKEN" \
     -F 'file=@leaf.jpg' \
     'http://localhost:8000/analyze?annotated=true' | jq
```

Rate-limit errors return 429 with a `Retry-After` header. Auth errors return 401 with a machine-readable `detail` string (`Missing bearer token`, `Token expired`, `Invalid token`, `Invalid authorization scheme`, `Token missing subject`).

## Tests

```bash
pytest
```
