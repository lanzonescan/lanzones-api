# Lanzones Scan API

A FastAPI service that detects lanzones crop conditions from an uploaded image using a YOLOv8s model. The initial model targets leaf conditions (`dried-leaf`, `healthy`, `leaf-rust`, `powdery-mildew`) and is designed to extend to insect and fruit detection as future models ship.

---

## Table of contents

1. [Overview](#overview)
2. [Tech stack](#tech-stack)
3. [How the system works](#how-the-system-works)
4. [Model internals](#model-internals)
5. [Local environment setup](#local-environment-setup)
6. [Environment variables](#environment-variables)
7. [API reference](#api-reference)
8. [Training](#training)
9. [Testing](#testing)
10. [GitHub setup](#github-setup)
11. [VPS setup (Dokploy host)](#vps-setup-dokploy-host)
12. [Dokploy deployment](#dokploy-deployment)
13. [Operations and troubleshooting](#operations-and-troubleshooting)

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

## Model internals

This section explains what the YOLOv8s model does to an image internally, both at training time (`train.py`) and at inference time (`inference.py`). Useful when tuning `imgsz`, `conf`, or interpreting training logs.

### How YOLO analyzes and processes an image

Setup recap: `YOLOv8s` pretrained on COCO, fine-tuned at `imgsz=640` on 4 classes (`dried-leaf`, `healthy`, `leaf-rust`, `powdery-mildew`) via Ultralytics. Every image goes through the following stages end-to-end.

#### 1. Preprocessing — image → tensor

For each image (training or inference):

1. **Letterbox resize to 640×640.** The image is scaled so its longer side = 640, then padded with gray (114, 114, 114) on the shorter side to keep aspect ratio. This avoids distortion that would warp leaf shapes.
2. **BGR → RGB**, then transposed from `H×W×C` to `C×H×W`.
3. **Normalize** pixel values `0..255 → 0..1` (float32, or float16 on MPS).
4. **Batch** into a `(B, 3, 640, 640)` tensor on `mps` (the `DEFAULT_DEVICE`).

During training only, Ultralytics also applies augmentations before step 1: Mosaic (4-image collage), HSV jitter, horizontal flip, scale/translate, and optionally MixUp. Mosaic teaches the model context at varied scales — useful for leaf disease that can appear as small patches.

#### 2. Backbone — feature extraction (CSPDarknet)

The 640×640 tensor flows through a CSPDarknet-style CNN that progressively downsamples:

```
640×640 → 320×320 → 160×160 → 80×80 → 40×40 → 20×20
 (stem)    (P1)      (P2)      (P3)    (P4)    (P5)
```

At each stage, `Conv → C2f → ...` blocks extract increasingly abstract features. Early layers see edges/color (good for rust spots vs. mildew dust); deep layers see whole-leaf shape and texture. The "s" in `yolov8s` means **small** — ~11M params, ~28 GFLOPs — a speed/accuracy trade chosen because the dataset and target hardware (MPS / mobile) are modest.

#### 3. Neck — multi-scale fusion (PAN-FPN)

The neck combines feature maps from P3 (80×80), P4 (40×40), P5 (20×20):

- **Top-down (FPN)**: upsample deep semantic features and add to shallower ones — so the 80×80 map knows *what* it's looking at, not just *where*.
- **Bottom-up (PAN)**: downsample and add back — so the 20×20 map keeps localization detail.

Three feature pyramids come out the other side: 80×80 detects **small** lesions, 40×40 detects **medium** patches, 20×20 detects **large** leaf-level signals.

#### 4. Head — anchor-free decoupled prediction

YOLOv8 uses a **decoupled, anchor-free** head. For every cell in each of the three pyramids:

- **Classification branch** → 4 logits (one per class in `CLASS_NAMES`).
- **Regression branch** → 4 distances (left, top, right, bottom) from the cell center to the predicted box edges, encoded as **DFL** (Distribution Focal Loss) — see below.

Total raw predictions per image: `80² + 40² + 20² = 8400` candidate boxes.

#### 5. Loss — what the model is being optimized for

During `model.train(...)`, three losses are summed per batch:

| Loss | Purpose | Function |
|---|---|---|
| `box` | Box localization | CIoU |
| `cls` | Class prediction | BCE with logits |
| `dfl` | Sub-pixel box refinement | Distribution Focal Loss |

Targets are matched to predictions via **TaskAlignedAssigner** (no anchors, no IoU thresholds — it picks the top-k cells whose joint cls-score × IoU is highest for each ground-truth box). This is what makes YOLOv8 anchor-free.

#### 6. Post-processing — raw outputs → final boxes

At inference (and during validation each epoch):

1. **Confidence filter**: drop boxes with `max class prob < conf` (the `DEFAULT_CONF = 0.25`).
2. **NMS** (non-max suppression, per class): keep the highest-confidence box in any cluster of overlapping boxes (IoU > 0.7 default).
3. **Letterbox-invert**: unscale and unpad boxes back to original image coordinates so you get pixel boxes on the real image.

Result: for each leaf in the image, one box + one class label + one confidence score — exactly the shape returned by `/analyze`.

#### 7. What `train.py` actually drives

- `YOLO('yolov8s.pt')` — load COCO-pretrained weights (the backbone + neck already know edges, textures, leaf-like shapes). Only the head's class predictor is reinitialized for 4 classes.
- `model.train(data=data_yaml, ...)` — runs the full loop above for `epochs=50`, validating each epoch, saving `last.pt` and `best.pt` (best = highest val mAP@0.5:0.95).
- `_find_best_weights` then copies `models/run/weights/best.pt` to `models/best.pt`, which the API serves.

#### Concretely for lanzones leaves

- A 4000×3000 phone photo of a leaf → letterboxed to 640×480 inside a 640×640 gray canvas.
- Backbone extracts texture: powdery-mildew shows up as a high-frequency white pattern on the P3 map; leaf-rust shows up as orange-channel blobs on P3/P4; dried-leaf is a low-saturation global signal that dominates P5.
- The head produces one box per leaf with a class distribution; NMS keeps one per leaf.
- Output to the Svelte client: `[{class: 'leaf-rust', conf: 0.87, box: [x1, y1, x2, y2]}, ...]`.

### DFL — Distribution Focal Loss

The trick that lets YOLOv8 predict box edges with **sub-pixel precision** without using anchors.

#### The problem it solves

A detection head has to predict 4 numbers per box: distances from the cell center to the **left, top, right, bottom** edges (call each one `d`).

The naive approach: regress `d` directly as a single float — e.g. "the left edge is 7.3 cells away." But:

- A single point estimate gives the model no way to express **uncertainty**. Is it 7.3 ± 0.1 (sharp edge) or 7.3 ± 2.0 (blurry, occluded edge)?
- Single-float regression with L1/L2 loss is notoriously **noisy near object boundaries** — the gradient is the same whether you're off by 0.5 px or 0.5 of a feature cell.
- Anchor-based heads dodged this by predicting offsets *from* an anchor — but YOLOv8 is anchor-free.

DFL fixes this by predicting a **discrete probability distribution** over possible distances, then taking its expectation.

#### The construction

Pick a max distance `reg_max` (YOLOv8 default: **16**). For each of the 4 edges, the head outputs **17 logits** — one per integer bucket `0, 1, 2, ..., 16`. Softmax those logits into a distribution `P(d = i)` for `i ∈ {0..16}`.

```
edge logits:  [l₀, l₁, l₂, ..., l₁₆]   ← 17 numbers per edge
softmax    →  [p₀, p₁, p₂, ..., p₁₆]   ← probabilities, sum to 1
expectation:  d̂ = Σ i · pᵢ              ← a single float, but continuous
```

Because `d̂` is a weighted average over integer buckets, it can land on **any real value** in `[0, 16]` — e.g. `d̂ = 7.3` arises naturally if probability mass clusters around buckets 7 and 8. That's the "sub-pixel" part.

So per box: `4 edges × 17 buckets = 68 numbers` are predicted (vs. just 4 in naive regression). For a feature-map cell at the 80×80 level, one bucket unit corresponds to one **stride** of 8 pixels on the original image — so `reg_max=16` means the head can describe edges up to 128 px from the cell center, with sub-pixel resolution everywhere in between.

#### The loss

The ground-truth distance for an edge is some real number `y` (e.g. `y = 7.3`). DFL turns this into a target over the two adjacent buckets:

```
y = 7.3  →  bucket 7 gets weight (8 - 7.3) = 0.7
            bucket 8 gets weight (7.3 - 7) = 0.3
            all others: 0
```

Then it applies **cross-entropy** against the predicted distribution at just those two buckets:

```
DFL = -[(y_high - y) · log(p_⌊y⌋)  +  (y - y_low) · log(p_⌈y⌉)]
    = -[ 0.7 · log(p₇)             +  0.3 · log(p₈)            ]
```

The "focal" name comes from the original paper (Generalized Focal Loss, Li et al. 2020) — it focuses learning on the two buckets that bracket the truth, leaving the model free to assign zero probability everywhere else.

#### Why this works better than direct regression

1. **Uncertainty is first-class.** A sharp leaf edge produces a tight spike in the distribution (`p₇ ≈ 1.0`). A fuzzy/occluded edge produces a wide hump (`p₆ ≈ 0.2, p₇ ≈ 0.5, p₈ ≈ 0.3`). The expected value is the same in both cases, but the **shape** is different — and downstream IoU-based losses can exploit it.
2. **Smooth gradients near integer boundaries.** With direct L1, a target of `7.5` is equidistant from any prediction in `[7, 8]` — the gradient is flat. With DFL, the target puts equal mass on buckets 7 and 8, and the cross-entropy gradient points the distribution toward exactly that split.
3. **Implicit regularization.** Forcing predictions through softmax + 17 buckets caps the loss landscape — you can't predict `d = 10000` and explode the gradient. `reg_max=16` bounds the expressible range to something physically reasonable.
4. **Works hand-in-hand with CIoU.** YOLOv8's box loss is `λ_box · CIoU + λ_dfl · DFL`. CIoU shapes the *overall* box (location, scale, aspect ratio); DFL sharpens *each individual edge*. They optimize complementary things.

#### At inference

The distribution is discarded and only the expectation is used:

```python
d_hat = sum(i * softmax(logits)[i] for i in range(reg_max + 1))  # one float per edge
```

then `(d_left, d_top, d_right, d_bottom)` are converted from "cell-stride units" back to pixel coordinates using the cell's `(cx, cy)` and the feature-map stride. The distribution itself is internal training machinery — runtime cost is just one softmax + one dot product per edge.

#### In the training logs

`model.train(...)` in `src/lanzonesscan/train.py` configures DFL from the YOLOv8s spec — there's nothing to tune directly. But the per-epoch log shows three numbers:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   ...
  1/50    2.3G       1.45        2.10       1.30
  ...
 50/50    2.3G       0.41        0.18       0.85
```

`dfl_loss` plateauing well above zero is normal — it measures distribution sharpness, not bounding-box correctness. The metric to actually watch is **mAP**, reported below those losses each validation epoch.

#### Tuning knob

If lanzones leaves are large and roughly centered (typical phone shots), the default `reg_max=16` × stride is plenty. If the task ever switched to detecting **small** disease spots tightly cropped on a leaf, lowering `reg_max` (e.g. 8) would give the distribution finer resolution within a smaller distance range — but that's a model architecture change, not a flag, so rarely worth it. Usually just train longer or bump `imgsz` above 640 instead.

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
| `PROXY_SECRET` | unset | no | If set, `/analyze` requires header `X-Proxy-Secret` to match. Used to let a Cloudflare WAF Skip rule trust traffic from the SvelteKit proxy. |
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
