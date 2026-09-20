"""The encoder and decoder must build identical ROI masks, or the composite tears."""
from __future__ import annotations

import unittest

import numpy as np

from faunacodec.compression.dual_stream_dcvc import _apply_roi_mask
from faunacodec.masking import build_frame_mask


class TestRoiMaskAgreement(unittest.TestCase):
    def test_transport_roi_box_map_is_accepted(self) -> None:
        mask = build_frame_mask(
            frame_idx=3,
            width=5,
            height=5,
            mask_source="frame_drop_roi_box",
            roi_boxes_map={},
            frame_drop_json={
                "roi_box_by_frame": {"3": {"x1": 1, "y1": 1, "x2": 3, "y2": 3}}
            },
            roi_min_conf=0.0,
            roi_dilate_px=0,
        )

        self.assertEqual(int(mask.sum()), 9 * 255)

    def test_apply_roi_mask_matches_decompression_mask_semantics(self) -> None:
        frame = np.arange(5 * 5 * 3, dtype=np.uint8).reshape(5, 5, 3)
        boxes = [{"x1": 1, "y1": 1, "x2": 3, "y2": 3}]

        expected_mask = build_frame_mask(
            frame_idx=0,
            width=frame.shape[1],
            height=frame.shape[0],
            mask_source="roi_detection",
            roi_boxes_map={0: boxes},
            frame_drop_json={},
            roi_min_conf=0.0,
            roi_dilate_px=0,
        )
        expected = frame.copy()
        expected[expected_mask == 0] = 0

        actual = _apply_roi_mask(frame, boxes, roi_min_conf=0.0)

        np.testing.assert_array_equal(actual, expected)

    def test_apply_roi_mask_honors_min_conf_without_visible_dilation(self) -> None:
        frame = np.arange(5 * 5 * 3, dtype=np.uint8).reshape(5, 5, 3)
        boxes = [{"x1": 1, "y1": 1, "x2": 3, "y2": 3, "conf": 0.1}]

        expected_mask = build_frame_mask(
            frame_idx=0,
            width=frame.shape[1],
            height=frame.shape[0],
            mask_source="roi_detection",
            roi_boxes_map={0: boxes},
            frame_drop_json={},
            roi_min_conf=0.5,
            roi_dilate_px=0,
        )
        expected = frame.copy()
        expected[expected_mask == 0] = 0

        actual = _apply_roi_mask(frame, boxes, roi_min_conf=0.5)

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
