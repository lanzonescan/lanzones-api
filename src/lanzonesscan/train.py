from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

from lanzonesscan.config import (
	DEFAULT_DEVICE,
	DEFAULT_EPOCHS,
	DEFAULT_IMGSZ,
	MODEL_PATH,
	MODELS_DIR,
	PRETRAINED_WEIGHTS
)
from lanzonesscan.data_setup import ensure_dataset


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description='Train YOLOv8s on the lanzones leaf dataset')
	p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
	p.add_argument('--device', default=DEFAULT_DEVICE, help='mps | cpu | cuda:0 | ...')
	p.add_argument('--imgsz', type=int, default=DEFAULT_IMGSZ)
	p.add_argument('--resume', action='store_true')
	return p.parse_args()


def main() -> None:
	args = parse_args()
	data_yaml = ensure_dataset()
	MODELS_DIR.mkdir(parents=True, exist_ok=True)

	model = YOLO(PRETRAINED_WEIGHTS)
	results = model.train(
		data=str(data_yaml),
		epochs=args.epochs,
		imgsz=args.imgsz,
		device=args.device,
		project=str(MODELS_DIR),
		name='run',
		exist_ok=True,
		resume=args.resume
	)

	best = _find_best_weights(results)
	if best is None:
		raise RuntimeError('Training finished but best.pt was not found')
	shutil.copy2(best, MODEL_PATH)
	print(f'Saved best weights to {MODEL_PATH}')


def _find_best_weights(results: Any) -> Path | None:
	save_dir = Path(getattr(results, 'save_dir', MODELS_DIR / 'run'))
	candidate = save_dir / 'weights' / 'best.pt'
	return candidate if candidate.exists() else None


if __name__ == '__main__':
	main()
