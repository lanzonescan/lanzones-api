import os

os.environ.setdefault('JWT_SECRET', 'test-secret-not-for-prod')

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tiny_zip(tmp_path: Path) -> Path:
	'''Build a minimal YOLO26-shaped zip mirroring the real dataset layout.'''
	zip_path = tmp_path / 'tiny.zip'
	with zipfile.ZipFile(zip_path, 'w') as z:
		z.writestr(
			'data.yaml',
			'train: ../train/images\nval: ../valid/images\ntest: ../test/images\n'
			"nc: 4\nnames: ['dried-leaf', 'healthy', 'leaf-rust', 'powdery-mildew']\n"
		)
		for split in ('train', 'valid', 'test'):
			z.writestr(f'{split}/images/sample.jpg', _tiny_jpeg())
			z.writestr(f'{split}/labels/sample.txt', '0 0.5 0.5 0.2 0.2\n')
	return zip_path


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
	return _tiny_jpeg()


def _tiny_jpeg() -> bytes:
	buf = io.BytesIO()
	Image.new('RGB', (64, 64), color=(120, 180, 90)).save(buf, format='JPEG')
	return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_rate_limits():
	from lanzonesscan.rate_limit import limiter
	limiter.reset()
	yield
	limiter.reset()


@pytest.fixture
def authed_override():
	from fastapi import Request
	from lanzonesscan.api import app
	from lanzonesscan.auth import get_current_subject

	def _override(request: Request) -> str:
		request.state.subject = 'test-user'
		return 'test-user'

	app.dependency_overrides[get_current_subject] = _override
	yield
	app.dependency_overrides.pop(get_current_subject, None)
