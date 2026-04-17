from pathlib import Path

import pytest

from lanzonesscan import data_setup


def test_ensure_dataset_extracts_and_rewrites_yaml(tmp_path: Path, tiny_zip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	data_dir = tmp_path / 'data'
	monkeypatch.setattr(data_setup, 'ZIP_PATH', tiny_zip)
	monkeypatch.setattr(data_setup, 'DATA_DIR', data_dir)

	result = data_setup.ensure_dataset()

	assert result == data_dir / 'data.yaml'
	assert (data_dir / 'train' / 'images' / 'sample.jpg').exists()
	content = result.read_text()
	assert 'train: train/images' in content
	assert 'val: valid/images' in content
	assert '../' not in content
	assert "'dried-leaf'" in content
	assert "'powdery-mildew'" in content


def test_ensure_dataset_is_idempotent(tmp_path: Path, tiny_zip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	data_dir = tmp_path / 'data'
	monkeypatch.setattr(data_setup, 'ZIP_PATH', tiny_zip)
	monkeypatch.setattr(data_setup, 'DATA_DIR', data_dir)

	data_setup.ensure_dataset()
	marker = data_dir / 'marker.txt'
	marker.write_text('unchanged')

	data_setup.ensure_dataset()

	assert marker.read_text() == 'unchanged'


def test_ensure_dataset_missing_zip_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(data_setup, 'ZIP_PATH', tmp_path / 'nope.zip')
	monkeypatch.setattr(data_setup, 'DATA_DIR', tmp_path / 'data')

	with pytest.raises(FileNotFoundError, match='nope.zip'):
		data_setup.ensure_dataset()
