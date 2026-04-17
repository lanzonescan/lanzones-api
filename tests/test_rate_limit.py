from unittest.mock import MagicMock

from lanzonesscan.rate_limit import key_by_ip, key_by_sub, limiter


def test_limiter_exists():
	assert limiter is not None


def test_key_by_sub_reads_request_state():
	req = MagicMock()
	req.state.subject = 'user-xyz'
	assert key_by_sub(req) == 'user-xyz'


def test_key_by_sub_falls_back_to_anonymous_when_no_subject():
	req = MagicMock(spec=['state'])
	req.state = type('S', (), {})()
	assert key_by_sub(req) == 'anonymous'


def test_key_by_ip_uses_client_host():
	req = MagicMock()
	req.client.host = '203.0.113.9'
	req.headers = {}
	assert key_by_ip(req) == '203.0.113.9'
