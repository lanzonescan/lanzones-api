import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from lanzonesscan import config
from lanzonesscan.config import MODEL_PATH
from lanzonesscan.inference import Detection, LeafDetector


def _make_mock_yolo_result(
	boxes_xyxy: list[list[float]],
	class_ids: list[float],
	confs: list[float],
	orig_shape: tuple[int, int] = (64, 64)
) -> MagicMock:
	mock_boxes = MagicMock()
	mock_boxes.xyxy = MagicMock()
	mock_boxes.xyxy.cpu.return_value.tolist.return_value = boxes_xyxy
	mock_boxes.cls = MagicMock()
	mock_boxes.cls.cpu.return_value.tolist.return_value = class_ids
	mock_boxes.conf = MagicMock()
	mock_boxes.conf.cpu.return_value.tolist.return_value = confs

	mock_result = MagicMock()
	mock_result.boxes = mock_boxes
	mock_result.orig_shape = orig_shape
	mock_result.names = {i: n for i, n in enumerate(config.CLASS_NAMES)}
	return mock_result


def test_detector_init_missing_weights_raises(tmp_path: Path):
	missing = tmp_path / 'nope.pt'
	with pytest.raises(FileNotFoundError, match='nope.pt'):
		LeafDetector(missing)


@patch('lanzonesscan.inference.YOLO')
def test_detector_init_loads_model(mock_yolo_cls: MagicMock, tmp_path: Path) -> None:
	weights = tmp_path / 'best.pt'
	weights.write_bytes(b'fake')
	LeafDetector(weights)
	mock_yolo_cls.assert_called_once_with(str(weights))


@patch('lanzonesscan.inference.YOLO')
def test_predict_returns_structured_detections(mock_yolo_cls: MagicMock, tmp_path: Path, sample_jpeg_bytes: bytes) -> None:
	weights = tmp_path / 'best.pt'
	weights.write_bytes(b'fake')
	mock_model = MagicMock()
	mock_model.predict.return_value = [
		_make_mock_yolo_result(
			boxes_xyxy=[[10.0, 20.0, 50.0, 60.0]],
			class_ids=[2.0],
			confs=[0.87]
		)
	]
	mock_yolo_cls.return_value = mock_model

	detector = LeafDetector(weights)
	detections = detector.predict(sample_jpeg_bytes, conf=0.25)

	assert isinstance(detections, list)
	assert len(detections) == 1
	d: Detection = detections[0]
	assert d['class'] == 'leaf-rust'
	assert d['confidence'] == pytest.approx(0.87)
	assert d['bbox'] == [10.0, 20.0, 50.0, 60.0]


@patch('lanzonesscan.inference.YOLO')
def test_predict_empty_results(mock_yolo_cls: MagicMock, tmp_path: Path, sample_jpeg_bytes: bytes) -> None:
	weights = tmp_path / 'best.pt'
	weights.write_bytes(b'fake')
	mock_model = MagicMock()
	mock_model.predict.return_value = [
		_make_mock_yolo_result(boxes_xyxy=[], class_ids=[], confs=[])
	]
	mock_yolo_cls.return_value = mock_model

	detector = LeafDetector(weights)
	assert detector.predict(sample_jpeg_bytes) == []


@patch('lanzonesscan.inference.YOLO')
def test_annotate_returns_valid_png_with_drawn_box(mock_yolo_cls: MagicMock, tmp_path: Path, sample_jpeg_bytes: bytes) -> None:
	weights = tmp_path / 'best.pt'
	weights.write_bytes(b'fake')
	mock_yolo_cls.return_value = MagicMock()

	detector = LeafDetector(weights)
	detections: list[Detection] = [
		{'class': 'leaf-rust', 'confidence': 0.9, 'bbox': [5.0, 5.0, 40.0, 40.0]}
	]
	png_bytes = detector.annotate(sample_jpeg_bytes, detections)

	img = Image.open(io.BytesIO(png_bytes))
	assert img.format == 'PNG'
	assert img.size == (64, 64)

	blank = Image.open(io.BytesIO(detector.annotate(sample_jpeg_bytes, [])))
	assert img.tobytes() != blank.tobytes()


def test_predict_invalid_image_raises(tmp_path: Path):
	weights = tmp_path / 'best.pt'
	weights.write_bytes(b'fake')
	with patch('lanzonesscan.inference.YOLO'):
		detector = LeafDetector(weights)
		with pytest.raises(ValueError, match='Invalid image'):
			detector.predict(b'not an image', conf=0.25)


@pytest.mark.skipif(
	not MODEL_PATH.exists(),
	reason='Real model weights missing — run `python -m lanzonesscan.train` to enable this test'
)
def test_predict_against_real_model(sample_jpeg_bytes: bytes):
	detector = LeafDetector(MODEL_PATH)
	detections = detector.predict(sample_jpeg_bytes, conf=0.01)
	assert isinstance(detections, list)
	for d in detections:
		assert set(d.keys()) == {'class', 'confidence', 'bbox'}
		assert isinstance(d['class'], str)
		assert 0.0 <= d['confidence'] <= 1.0
		assert len(d['bbox']) == 4
