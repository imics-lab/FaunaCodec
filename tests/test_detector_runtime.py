from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from faunacodec.detection.detector import _detect_proc_boxes, _detector_quantize


class _Model:
    def __init__(self) -> None:
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [SimpleNamespace(boxes=None)]


def test_detector_uses_current_ultralytics_quantize_keyword():
    model = _Model()

    boxes, scores = _detect_proc_boxes(
        model=model,
        frame_p_bgr=np.zeros((12, 16, 3), dtype=np.uint8),
        imgsz=640,
        conf_thr=0.25,
        iou_thr=0.5,
        device="cuda:0",
        quantize=16,
        target_class_ids={0},
        bbox_pad_frac=0.0,
        bbox_pad_px=0.0,
    )

    assert boxes == []
    assert scores == []
    assert model.kwargs["quantize"] == 16
    assert "half" not in model.kwargs


def test_legacy_half_config_maps_to_quantize_without_changing_precision():
    assert _detector_quantize({"half": True}) == 16
    assert _detector_quantize({"half": False}) is None
    assert _detector_quantize({"quantize": 32, "half": True}) == 32
