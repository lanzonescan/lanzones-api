from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import jwt

from lanzonesscan import config

security = HTTPBearer(auto_error=False)


def decode_token(token: str) -> dict[str, Any]:
	if not config.JWT_SECRET:
		raise HTTPException(status_code=500, detail='JWT secret not configured')
	try:
		return jwt.decode(
			token,
			config.JWT_SECRET,
			algorithms=[config.JWT_ALGORITHM],
			leeway=config.JWT_LEEWAY_SECONDS,
			issuer=config.JWT_ISSUER,
			audience=config.JWT_AUDIENCE,
			options={'require': ['exp']}
		)
	except jwt.ExpiredSignatureError:
		raise HTTPException(status_code=401, detail='Token expired')
	except jwt.InvalidTokenError:
		raise HTTPException(status_code=401, detail='Invalid token')


def get_current_subject(
	request: Request,
	credentials: HTTPAuthorizationCredentials | None = Depends(security)
) -> str:
	if credentials is None:
		raise HTTPException(status_code=401, detail='Missing bearer token')
	if credentials.scheme.lower() != 'bearer':
		raise HTTPException(status_code=401, detail='Invalid authorization scheme')

	payload = decode_token(credentials.credentials)
	sub = payload.get('sub')
	if not sub or not isinstance(sub, str):
		raise HTTPException(status_code=401, detail='Token missing subject')

	request.state.subject = sub
	return sub
