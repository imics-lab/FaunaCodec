# Package internals

Read this if you are modifying the pipeline. For using it, see the [top-level README](../README.md).

## Where each module runs

The archive is the only interface between the two halves, so each side can be deployed and upgraded independently.

| Edge device (Jetson-class) | Server (cloud GPU) |
|---|---|
| `detection/` — MegaDetector V6 + KLT tracking | `decompression/` — stream decoding, AMT interpolation, compositing |
| `selection/` — motion state machine, dual frame schedule | `restoration/` — RestoreUNet artifact removal (optional) |
| `masking/` — boxes to pixel masks | `upscaling/` — 2x delivery (optional) |
| `compression/` — dual-stream encoding | |
| `archive.py` — writes the archive | `archive.py` — reads the archive |

`stages.py` is the public API both sides call; `cli/` wraps it in argparse; `config.py` validates a config before an expensive run; `paths.py` resolves the paths a config names; `video_io.py` is the only place that touches video files.

## What flows between stages

Every stage takes and returns plain data, so any one of them can be replaced or tested in isolation.

```
detection    video path          -> {"frames": {frame_idx: [box, ...]}, ...}
selection    video path, boxes   -> {"roi_kept_frames": [...], "bg_kept_frames": [...], ...}
compression  video, boxes, keep  -> {"roi_bin_bytes", "bg_bin_bytes", "meta"}
archive      the above           -> archive bytes
decompression archive            -> (frames, fps, width, height, detections)
restoration  frames              -> frames
upscaling    frames              -> frames
```

A box is a dict with `x1`, `y1`, `x2`, `y2` and usually `conf` and a track id. `masking.boxes_for_frame` accepts either int or str frame keys, because the boxes make a round trip through JSON inside the archive.

## Adding a codec backend

`compression/__init__.py` dispatches on `compression.codec`. A backend is one function:

```python
def encode_roi_bg_<name>(*, source_video_path, roi_bbox_map, frame_drop_result,
                         compression_cfg, root_dir) -> dict
```

returning `{"roi_bin_bytes", "bg_bin_bytes", "meta"}`. The frame iteration and ROI masking helpers in `compression/dual_stream_dcvc.py` are codec-agnostic and are what `dual_stream_ffmpeg.py` reuses. The decoder side needs a matching entry in `decompression/roi_bg_decompress.py`.

## Adding an upscaling backend

Write a class taking a config dict with an `upscale_sequence(frames, detections=None) -> list[np.ndarray]` method, then register it in `upscaling/__init__.py`'s `UPSCALER_BACKENDS`. Backends are imported lazily because several pull in mutually incompatible versions of `diffusers`. `detections` is passed so a backend can leave the subject region alone; the per-frame backends ignore it.

## Things that will bite you

- **`compression/dcvc_encoder.py` imports `src.models`, which is DCVC's package**, not this one. It puts the vendored `third_party/dcvc/` checkout at the front of `sys.path` first. This is why this package is called `faunacodec` and not `src`.
- **DCVC quantization parameters run opposite to CRF.** Higher `qp` is better quality; lower `crf` is better quality. `configs/compression.yaml` carries both.
- **Detection and encoding may use different resolutions, but their timelines must match.** The CLIs automatically retain the original video for detection when they resize for encoding. Python callers pass it as `detection_video_path`; `stages.compress` scales the boxes and rejects mismatched frame counts or frame rates.
- **Float DCVC is not portable across CPU architectures.** See the top-level README.
