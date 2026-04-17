from __future__ import annotations

import io
from pathlib import Path
from typing import Any, TypedDict

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

from lanzonesscan.config import CLASS_NAMES, DEFAULT_CONF, DEFAULT_IMGSZ

Detection = TypedDict('Detection', {'class': str, 'confidence': float, 'bbox': list[float]})

_BOX_COLORS = {
	'dried-leaf': (210, 150, 60),
	'healthy': (60, 180, 80),
	'leaf-rust': (200, 60, 60),
	'powdery-mildew': (180, 180, 220)
}


class LeafDetector:
	'''YOLO wrapper. Not thread-safe — callers must serialize access to predict().'''
	def __init__(self, model_path: Path):
		if not Path(model_path).exists():
			raise FileNotFoundError(
				f'Model weights not found at {model_path}. '
				'Run `python -m lanzonesscan.train` to produce them.'
			)
		self.model_path = Path(model_path)
		self.model = YOLO(str(model_path))

	def predict(self, image_bytes: bytes, conf: float = DEFAULT_CONF) -> list[Detection]:
		image = self._load_image(image_bytes)
		results = self.model.predict(
			image,
			conf=conf,
			imgsz=DEFAULT_IMGSZ,
			verbose=False
		)
		return self._results_to_detections(results)

	def annotate(self, image_bytes: bytes, detections: list[Detection]) -> bytes:
		image = self._load_image(image_bytes).convert('RGB')
		draw = ImageDraw.Draw(image)
		font = self._load_font()
		for d in detections:
			x1, y1, x2, y2 = d['bbox']
			color = _BOX_COLORS.get(d['class'], (255, 255, 255))
			draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
			label = f"{d['class']} {d['confidence']:.2f}"
			draw.text((x1 + 4, max(y1 - 14, 0)), label, fill=color, font=font)
		buf = io.BytesIO()
		image.save(buf, format='PNG')
		return buf.getvalue()

	@staticmethod
	def _load_image(image_bytes: bytes) -> Image.Image:
		try:
			img = Image.open(io.BytesIO(image_bytes))
			img.load()
			return img
		except (UnidentifiedImageError, OSError) as e:
			raise ValueError(f'Invalid image: {e}') from e

	@staticmethod
	def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
		try:
			return ImageFont.truetype('Arial.ttf', 12)
		except (OSError, IOError):
			return ImageFont.load_default(size=12)

	@staticmethod
	def _results_to_detections(results: list[Any]) -> list[Detection]:
		detections: list[Detection] = []
		for r in results:
			if r.boxes is None:
				continue
			xyxy = r.boxes.xyxy.cpu().tolist()
			if len(xyxy) == 0:
				continue
			cls_ids = r.boxes.cls.cpu().tolist()
			confs = r.boxes.conf.cpu().tolist()
			names = getattr(r, 'names', None) or {i: n for i, n in enumerate(CLASS_NAMES)}
			for box, cid, c in zip(xyxy, cls_ids, confs, strict=True):
				detections.append({
					'class': names[int(cid)],
					'confidence': float(c),
					'bbox': [float(v) for v in box]
				})
		return detections
