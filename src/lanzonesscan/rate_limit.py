from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from lanzonesscan.config import RATE_LIMIT_STORAGE_URI


def key_by_ip(request: Request) -> str:
	return get_remote_address(request)


def key_by_sub(request: Request) -> str:
	return getattr(request.state, 'subject', 'anonymous')


limiter = Limiter(key_func=key_by_ip, storage_uri=RATE_LIMIT_STORAGE_URI)


def rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
	if not isinstance(exc, RateLimitExceeded):
		raise exc
	retry_after = getattr(exc, 'retry_after', 60)
	response = JSONResponse(
		status_code=429,
		content={'detail': f'Rate limit exceeded: {exc.detail}'}
	)
	response.headers['Retry-After'] = str(retry_after)
	return response
