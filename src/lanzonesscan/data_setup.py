from __future__ import annotations

import zipfile
from pathlib import Path

from lanzonesscan.config import CLASS_NAMES, DATA_DIR, ZIP_PATH


def ensure_dataset() -> Path:
	yaml_path = DATA_DIR / 'data.yaml'
	if yaml_path.exists():
		return yaml_path
	if not ZIP_PATH.exists():
		raise FileNotFoundError(f'Dataset zip not found at {ZIP_PATH}')
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	with zipfile.ZipFile(ZIP_PATH) as z:
		z.extractall(DATA_DIR)
	_write_data_yaml(yaml_path)
	return yaml_path


def _write_data_yaml(path: Path) -> None:
	names_literal = '[' + ', '.join(f"'{n}'" for n in CLASS_NAMES) + ']'
	content = (
		f'path: {DATA_DIR.resolve()}\n'
		f'train: train/images\n'
		f'val: valid/images\n'
		f'test: test/images\n'
		f'nc: {len(CLASS_NAMES)}\n'
		f'names: {names_literal}\n'
	)
	path.write_text(content)
