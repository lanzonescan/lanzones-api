import io
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lanzonesscan import api as api_module
from lanzonesscan.api import app, get_detector


@pytest.fixture
def mock_detector() -> MagicMock:
	detector = MagicMock()
	detector.predict.return_value = [
		{'class': 'leaf-rust', 'confidence': 0.9, 'bbox': [10.0, 10.0, 40.0, 40.0]}
	]
	detector.annotate.return_value = _png_bytes()
	return detector


@pytest.fixture
def client(mock_detector: MagicMock, authed_override: None) -> Iterator[TestClient]:
	app.dependency_overrides[get_detector] = lambda: mock_detector
	app.state.detector = mock_detector
	with TestClient(app) as c:
		yield c
	app.dependency_overrides.pop(get_detector, None)


def _png_bytes() -> bytes:
	buf = io.BytesIO()
	Image.new('RGB', (64, 64), color=(10, 20, 30)).save(buf, format='PNG')
	return buf.getvalue()


def _jpeg_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
	buf = io.BytesIO()
	Image.new('RGB', size, color=(100, 160, 80)).save(buf, format='JPEG')
	return buf.getvalue()


def test_health(client: TestClient):
	r = client.get('/health')
	assert r.status_code == 200
	body = r.json()
	assert body['status'] == 'ok'
	assert body['model_loaded'] is True


def test_analyze_returns_detections(client: TestClient):
	r = client.post(
		'/analyze',
		files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
	)
	assert r.status_code == 200
	body = r.json()
	assert body['filename'] == 'leaf.jpg'
	assert body['image_size'] == {'width': 64, 'height': 64}
	assert body['annotated_image'] is None
	assert len(body['detections']) == 1
	d = body['detections'][0]
	assert d['class'] == 'leaf-rust'
	assert d['confidence'] == pytest.approx(0.9)
	assert d['bbox'] == [10.0, 10.0, 40.0, 40.0]


def test_analyze_with_annotated_true(client: TestClient):
	r = client.post(
		'/analyze?annotated=true',
		files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
	)
	assert r.status_code == 200
	body = r.json()
	assert body['annotated_image'] is not None
	assert body['annotated_image'].startswith('data:image/png;base64,')


def test_analyze_rejects_wrong_content_type(client: TestClient):
	r = client.post(
		'/analyze',
		files={'file': ('notes.txt', b'hello', 'text/plain')}
	)
	assert r.status_code == 415


def test_analyze_rejects_oversized(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(api_module, 'MAX_UPLOAD_BYTES', 100)
	r = client.post(
		'/analyze',
		files={'file': ('big.jpg', _jpeg_bytes(size=(256, 256)), 'image/jpeg')}
	)
	assert r.status_code == 413


def test_analyze_rejects_corrupt_image(client: TestClient):
	r = client.post(
		'/analyze',
		files={'file': ('bad.jpg', b'not-an-image', 'image/jpeg')}
	)
	assert r.status_code == 400


def test_analyze_rejects_invalid_conf(client: TestClient):
	r = client.post(
		'/analyze?conf=1.5',
		files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
	)
	assert r.status_code == 422


import time

import jwt as pyjwt
from lanzonesscan import config as app_config


def _sign(payload: dict[str, Any]) -> str:
	return pyjwt.encode(payload, app_config.JWT_SECRET, algorithm=app_config.JWT_ALGORITHM)


def _valid_token(sub: str = 'integration-user') -> str:
	return _sign({'sub': sub, 'exp': int(time.time()) + 60})


@pytest.fixture
def unauthed_client(mock_detector: MagicMock) -> Iterator[TestClient]:
	app.state.detector = mock_detector
	app.dependency_overrides[get_detector] = lambda: mock_detector
	with TestClient(app) as c:
		yield c
	app.dependency_overrides.pop(get_detector, None)


def test_health_still_open_without_auth(unauthed_client: TestClient):
	r = unauthed_client.get('/health')
	assert r.status_code == 200


def test_analyze_without_auth_returns_401(unauthed_client: TestClient):
	r = unauthed_client.post(
		'/analyze',
		files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
	)
	assert r.status_code == 401
	assert r.json()['detail'] == 'Missing bearer token'


def test_analyze_with_expired_token_returns_401(unauthed_client: TestClient):
	expired = _sign({'sub': 'u', 'exp': int(time.time()) - 60})
	r = unauthed_client.post(
		'/analyze',
		headers={'Authorization': f'Bearer {expired}'},
		files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
	)
	assert r.status_code == 401
	assert r.json()['detail'] == 'Token expired'


def test_analyze_with_token_missing_sub_returns_401(unauthed_client: TestClient):
	bad = _sign({'exp': int(time.time()) + 60})
	r = unauthed_client.post(
		'/analyze',
		headers={'Authorization': f'Bearer {bad}'},
		files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
	)
	assert r.status_code == 401
	assert r.json()['detail'] == 'Token missing subject'


def test_analyze_with_valid_token_returns_200(unauthed_client: TestClient):
	token = _valid_token()
	r = unauthed_client.post(
		'/analyze',
		headers={'Authorization': f'Bearer {token}'},
		files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
	)
	assert r.status_code == 200


def test_sub_rate_limit_triggers_after_11th_request(unauthed_client: TestClient):
	token = _valid_token(sub='user-burst')
	for _ in range(10):
		r = unauthed_client.post(
			'/analyze',
			headers={'Authorization': f'Bearer {token}'},
			files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
		)
		assert r.status_code == 200
	r = unauthed_client.post(
		'/analyze',
		headers={'Authorization': f'Bearer {token}'},
		files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
	)
	assert r.status_code == 429
	assert 'Retry-After' in r.headers


def test_ip_rate_limit_triggers_after_31st_request(unauthed_client: TestClient):
	for i in range(30):
		token = _valid_token(sub=f'user-{i}')
		r = unauthed_client.post(
			'/analyze',
			headers={'Authorization': f'Bearer {token}'},
			files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
		)
		assert r.status_code == 200, f'request {i} should have succeeded'
	token = _valid_token(sub='user-31')
	r = unauthed_client.post(
		'/analyze',
		headers={'Authorization': f'Bearer {token}'},
		files={'file': ('leaf.jpg', _jpeg_bytes(), 'image/jpeg')}
	)
	assert r.status_code == 429
	assert 'Retry-After' in r.headers
