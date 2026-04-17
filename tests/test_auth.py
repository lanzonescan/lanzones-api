import time
from typing import Any
from unittest.mock import MagicMock

import jwt
import pytest
from fastapi import HTTPException

from lanzonesscan import config
from lanzonesscan.auth import decode_token, get_current_subject


def _make_token(payload: dict[str, Any], secret: str | None = None, algorithm: str | None = None) -> str:
	return jwt.encode(
		payload,
		secret if secret is not None else config.JWT_SECRET,
		algorithm=algorithm if algorithm is not None else config.JWT_ALGORITHM
	)


def _request_with_state() -> MagicMock:
	req = MagicMock()
	req.state = MagicMock()
	return req


def _creds(token: str):
	from fastapi.security import HTTPAuthorizationCredentials
	return HTTPAuthorizationCredentials(scheme='Bearer', credentials=token)


def test_decode_token_valid():
	token = _make_token({'sub': 'user-123', 'exp': int(time.time()) + 60})
	payload = decode_token(token)
	assert payload['sub'] == 'user-123'


def test_decode_token_expired_raises_401_with_expired_detail():
	token = _make_token({'sub': 'user-123', 'exp': int(time.time()) - 60})
	with pytest.raises(HTTPException) as exc:
		decode_token(token)
	assert exc.value.status_code == 401
	assert exc.value.detail == 'Token expired'


def test_decode_token_wrong_secret_raises_invalid_token():
	token = _make_token({'sub': 'user-123', 'exp': int(time.time()) + 60}, secret='wrong-secret')
	with pytest.raises(HTTPException) as exc:
		decode_token(token)
	assert exc.value.status_code == 401
	assert exc.value.detail == 'Invalid token'


def test_decode_token_none_alg_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
	token = jwt.encode({'sub': 'user-123'}, '', algorithm='none')
	with pytest.raises(HTTPException) as exc:
		decode_token(token)
	assert exc.value.status_code == 401
	assert exc.value.detail == 'Invalid token'


def test_decode_token_issuer_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(config, 'JWT_ISSUER', 'https://auth.example.com')
	token = _make_token({'sub': 'user-123', 'iss': 'https://evil.example.com', 'exp': int(time.time()) + 60})
	with pytest.raises(HTTPException) as exc:
		decode_token(token)
	assert exc.value.status_code == 401
	assert exc.value.detail == 'Invalid token'


def test_decode_token_audience_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(config, 'JWT_AUDIENCE', 'lanzonesscan')
	token = _make_token({'sub': 'user-123', 'aud': 'other-service', 'exp': int(time.time()) + 60})
	with pytest.raises(HTTPException) as exc:
		decode_token(token)
	assert exc.value.status_code == 401
	assert exc.value.detail == 'Invalid token'


def test_get_current_subject_missing_credentials_raises_missing_bearer():
	req = _request_with_state()
	with pytest.raises(HTTPException) as exc:
		get_current_subject(req, None)
	assert exc.value.status_code == 401
	assert exc.value.detail == 'Missing bearer token'


def test_get_current_subject_wrong_scheme_raises_invalid_scheme():
	from fastapi.security import HTTPAuthorizationCredentials
	req = _request_with_state()
	creds = HTTPAuthorizationCredentials(scheme='Basic', credentials='abc')
	with pytest.raises(HTTPException) as exc:
		get_current_subject(req, creds)
	assert exc.value.status_code == 401
	assert exc.value.detail == 'Invalid authorization scheme'


def test_get_current_subject_missing_sub_raises():
	token = _make_token({'exp': int(time.time()) + 60})
	req = _request_with_state()
	with pytest.raises(HTTPException) as exc:
		get_current_subject(req, _creds(token))
	assert exc.value.status_code == 401
	assert exc.value.detail == 'Token missing subject'


def test_get_current_subject_empty_sub_raises():
	token = _make_token({'sub': '', 'exp': int(time.time()) + 60})
	req = _request_with_state()
	with pytest.raises(HTTPException) as exc:
		get_current_subject(req, _creds(token))
	assert exc.value.status_code == 401
	assert exc.value.detail == 'Token missing subject'


def test_get_current_subject_valid_stashes_subject_on_request_state():
	token = _make_token({'sub': 'user-42', 'exp': int(time.time()) + 60})
	req = _request_with_state()
	result = get_current_subject(req, _creds(token))
	assert result == 'user-42'
	assert req.state.subject == 'user-42'
