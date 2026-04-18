from __future__ import annotations

import base64
import io
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from slowapi.errors import RateLimitExceeded

from lanzonesscan import config
from lanzonesscan.auth import get_current_subject
from lanzonesscan.config import ACCEPTED_MIME, DEFAULT_CONF, MODEL_PATH
from lanzonesscan.inference import Detection, LanzonesDetector
from lanzonesscan.rate_limit import key_by_ip, key_by_sub, limiter, rate_limit_handler

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = config.MAX_UPLOAD_BYTES


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
	logging.basicConfig(level=logging.INFO)
	config.require_jwt_secret()
	if not getattr(app.state, 'detector', None):
		app.state.detector = LanzonesDetector(MODEL_PATH)
	yield


app = FastAPI(title='Lanzones Scan API', version='0.1.0', lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)


def get_detector(request: Request) -> LanzonesDetector:
	return request.app.state.detector


@app.get('/health')
def health(request: Request) -> dict[str, Any]:
	return {
		'status': 'ok',
		'model_loaded': getattr(request.app.state, 'detector', None) is not None
	}


@app.post('/analyze')
@limiter.limit(config.RATE_LIMIT_PER_IP, key_func=key_by_ip)
@limiter.limit(config.RATE_LIMIT_PER_SUB, key_func=key_by_sub)
def analyze(
	request: Request,
	file: UploadFile = File(...),
	conf: float = Query(DEFAULT_CONF, ge=0.0, le=1.0),
	annotated: bool = Query(False),
	subject: str = Depends(get_current_subject),
	detector: LanzonesDetector = Depends(get_detector)
) -> dict[str, Any]:
	if file.content_type not in ACCEPTED_MIME:
		raise HTTPException(status_code=415, detail=f'Unsupported content-type: {file.content_type}')

	image_bytes = file.file.read()
	if len(image_bytes) > MAX_UPLOAD_BYTES:
		raise HTTPException(status_code=413, detail='File exceeds maximum size')

	try:
		img = Image.open(io.BytesIO(image_bytes))
		img.load()
		width, height = img.size
	except (UnidentifiedImageError, OSError) as e:
		logger.warning('invalid image upload for %s: %s', file.filename, e)
		raise HTTPException(status_code=400, detail='Invalid image') from e

	started = time.perf_counter()
	try:
		detections: list[Detection] = detector.predict(image_bytes, conf=conf)
	except ValueError as e:
		logger.warning('predict rejected image for %s: %s', file.filename, e)
		raise HTTPException(status_code=400, detail='Invalid image') from e
	except Exception:
		logger.exception('inference failed for %s', file.filename)
		raise HTTPException(status_code=500, detail='Inference failed')

	elapsed_ms = (time.perf_counter() - started) * 1000
	logger.info(
		'analyze sub=%s file=%s size=%dx%d detections=%d duration_ms=%.1f',
		subject, file.filename, width, height, len(detections), elapsed_ms
	)

	annotated_b64 = None
	if annotated:
		png = detector.annotate(image_bytes, detections)
		annotated_b64 = 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')

	return {
		'filename': file.filename,
		'image_size': {'width': width, 'height': height},
		'detections': detections,
		'annotated_image': annotated_b64
	}
