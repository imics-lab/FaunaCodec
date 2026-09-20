"""Turning detection boxes into pixel masks and alpha mattes."""

from .masks import boxes_for_frame, build_boxes_mask, build_frame_mask, compose_soft, mask_to_alpha

__all__ = [
    "boxes_for_frame",
    "build_boxes_mask",
    "build_frame_mask",
    "compose_soft",
    "mask_to_alpha",
]
