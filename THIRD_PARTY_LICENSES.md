# Third-party components

FaunaCodec itself is Apache-2.0 (see `LICENSE`). The directories below are vendored snapshots of upstream projects, kept byte-for-byte so they can be diffed against their sources. Each remains under its own license.

| Directory | Upstream | License | Used for |
|---|---|---|---|
| `third_party/dcvc/` | [microsoft/DCVC](https://github.com/microsoft/DCVC) | MIT | Learned video codec (DMCI intra model, DMC inter model) and its C++ rANS entropy coder |
| `third_party/dcvc_int16/` | Derived from [microsoft/DCVC](https://github.com/microsoft/DCVC) | MIT | FaunaCodec-maintained integer-quantized DCVC runtime; see `third_party/dcvc_int16/NOTICE` |
| `third_party/amt/` | [MCG-NKU/AMT](https://github.com/MCG-NKU/AMT) | **CC BY-NC 4.0** | Frame interpolation for dropped background frames |
| `third_party/osediff/` | [cswry/OSEDiff](https://github.com/cswry/OSEDiff), bundling [recognize-anything](https://github.com/xinyu1205/recognize-anything) | Apache-2.0 | One-step diffusion upscaling backend, carrying local patches for newer `diffusers`/`peft` |

Fetched at runtime rather than vendored:

| Component | Source | License | Fetched by |
|---|---|---|---|
| S3Diff | [ArcticHare105/S3Diff](https://github.com/ArcticHare105/S3Diff) | Apache-2.0 | `faunacodec/upscaling/s3diff_upscaler.py` on first use |
| SD-Turbo | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) | Stability AI Community License | S3Diff backend |
| Stable Diffusion 2.1 base | [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) | CreativeML Open RAIL++-M | OSEDiff backend |
| PiD | [PiD release](https://github.com/Huage001/PiD) | see upstream | PiD backend, checkpoint supplied by the user |
| Wan2.1 / CogVideoX | Hugging Face | see upstream model cards | Video-native upscaling backends |

## Non-commercial restriction

`third_party/amt/` (AMT) is licensed **CC BY-NC 4.0**, which permits non-commercial use only. AMT interpolates the background frames the encoder dropped, so it sits on the default decompression path rather than in an optional stage. As bundled, this distribution is therefore usable for research and other non-commercial purposes only, notwithstanding FaunaCodec's own Apache-2.0 license.

Running `faunacodec-decompress --no-interpolate` (or `decompression.interpolate.enable: false`) holds each dropped frame instead of interpolating it, which avoids AMT entirely at some cost in temporal smoothness.

## Model weights

The weights published with this repository are redistributed under the license of the model they derive from:

- **MegaDetector V6 (MDV6-yolov9-c)** — [microsoft/CameraTraps](https://github.com/microsoft/CameraTraps), MIT. Note that the YOLOv9 architecture it builds on is GPL-3.0; the weights are redistributed here unmodified for research use.
- **DCVC CVPR-2025 intra/inter checkpoints** — microsoft/DCVC, MIT.
- **AMT-S / AMT-L** — MCG-NKU/AMT, MIT.
- **RestoreUNet checkpoints** (`restore_l1`, `restore_T1`, `restore_T5`) — trained for this work, Apache-2.0.
