import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
REPO_ROOT = PROJECT_ROOT.parent

ZIP_PATH = REPO_ROOT / 'Lanzones.v1i.yolo26.zip'
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
MODEL_PATH = Path(os.environ['MODEL_PATH']) if 'MODEL_PATH' in os.environ else MODELS_DIR / 'best.pt'

CLASS_NAMES = ['dried-leaf', 'healthy', 'leaf-rust', 'powdery-mildew']

DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ACCEPTED_MIME = frozenset({'image/jpeg', 'image/png', 'image/webp'})

PRETRAINED_WEIGHTS = 'yolov8s.pt'
DEFAULT_EPOCHS = 50
DEFAULT_DEVICE = 'mps'

PROXY_SECRET = os.environ.get('PROXY_SECRET') or None

JWT_SECRET = os.environ.get('JWT_SECRET')
JWT_ALGORITHM = os.environ.get('JWT_ALGORITHM', 'HS256')
JWT_LEEWAY_SECONDS = int(os.environ.get('JWT_LEEWAY_SECONDS', '0'))
JWT_ISSUER = os.environ.get('JWT_ISSUER') or None
JWT_AUDIENCE = os.environ.get('JWT_AUDIENCE') or None

RATE_LIMIT_STORAGE_URI = os.environ.get('RATE_LIMIT_STORAGE_URI', 'memory://')
RATE_LIMIT_PER_SUB = os.environ.get('RATE_LIMIT_PER_SUB', '10/minute')
RATE_LIMIT_PER_IP = os.environ.get('RATE_LIMIT_PER_IP', '30/minute')


def require_jwt_secret() -> str:
	if not JWT_SECRET:
		raise RuntimeError(
			'JWT_SECRET env var is required. '
			'Set it before starting the API: export JWT_SECRET=...'
		)
	return JWT_SECRET
