"""DCVC encoding, as a pure function of frames and config.

Loads no YAML and writes no files: `encode_dcvc_frames_to_bytes` takes an iterable of BGR frames and returns the bitstream plus the metadata needed to decode it, including the index map that puts the kept frames back on the original timeline.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..config import coerce_bool
from ..video_io import VideoInfo


def _setup_dcvc_imports(dcvc_repo_dir: str) -> Path:
    """Put the vendored DCVC checkout at the front of sys.path so `src.models` resolves to it."""
    repo = Path(dcvc_repo_dir).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"DCVC repo not found: {repo}")

    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))
    return repo


def _normalize_cuda_index(cuda_idx: object | None) -> int | None:
    if cuda_idx is None or isinstance(cuda_idx, bool):
        return None
    raw = cuda_idx[0] if isinstance(cuda_idx, (list, tuple)) and cuda_idx else cuda_idx
    try:
        idx = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid compression.dcvc.cuda_idx: {cuda_idx!r}") from exc
    if idx < 0:
        raise ValueError("compression.dcvc.cuda_idx must be >= 0")
    return idx


def _configure_cuda(use_cuda: bool, cuda_idx: object | None) -> torch.device:
    if not bool(use_cuda):
        raise ValueError("Strict GPU runtime forbids compression.dcvc.use_cuda=false.")
    if not torch.cuda.is_available():
        raise RuntimeError("Strict GPU runtime requires CUDA, but torch.cuda.is_available() is false.")
    selected_idx = _normalize_cuda_index(cuda_idx)
    if selected_idx is None:
        selected_idx = 0
    device_count = int(torch.cuda.device_count() or 0)
    if device_count > 0 and selected_idx >= device_count:
        raise ValueError(
            f"compression.dcvc.cuda_idx={selected_idx} is out of range for {device_count} visible CUDA device(s)."
        )
    torch.cuda.set_device(int(selected_idx))
    return torch.device(f"cuda:{int(selected_idx)}")


def bgr_to_yuv420_planes(frame_bgr: np.ndarray):
    """
    Same conversion as refs/compression.py.
    Returns:
      y: (1,H,W)
      uv: (2,H/2,W/2)
    """
    h, w = frame_bgr.shape[:2]
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
    flat = yuv.reshape(-1)
    y_size = w * h
    uv_size = (w * h) // 4
    y = flat[:y_size].reshape(1, h, w)
    u = flat[y_size:y_size + uv_size].reshape(h // 2, w // 2)
    v = flat[y_size + uv_size:y_size + 2 * uv_size].reshape(h // 2, w // 2)
    uv = np.stack([u, v], axis=0)
    return y, uv


def _encode_dcvc_core(
    *,
    frame_iter: Iterable[tuple[int, np.ndarray]],
    info: VideoInfo,
    cfg: dict[str, Any],
    video_path: str,
    use_kept_frames: bool,
) -> dict[str, Any]:
    dcvc_cfg = cfg.get("dcvc", {}) or {}
    q_cfg = cfg.get("quality", {}) or {}

    repo_dir = str(dcvc_cfg.get("repo_dir", "third_party/dcvc"))
    model_i = str(dcvc_cfg.get("model_i", ""))
    model_p = str(dcvc_cfg.get("model_p", ""))
    if not model_i or not model_p:
        raise ValueError("compression.dcvc.model_i and compression.dcvc.model_p must be set")

    force_zero_thres = dcvc_cfg.get("force_zero_thres", None)
    use_cuda = coerce_bool(dcvc_cfg.get("use_cuda", True), default=True)
    cuda_idx = dcvc_cfg.get("cuda_idx", None)
    reset_interval = int(dcvc_cfg.get("reset_interval", 32))
    intra_period = int(dcvc_cfg.get("intra_period", -1))
    max_frames = int(dcvc_cfg.get("max_frames", -1))

    qp_i = int(q_cfg.get("qp_i", 3))
    qp_p = int(q_cfg.get("qp_p", 3))

    _setup_dcvc_imports(repo_dir)
    device = _configure_cuda(use_cuda, cuda_idx)

    from src.layers.cuda_inference import replicate_pad  # noqa: E402
    from src.models.image_model import DMCI  # noqa: E402
    from src.utils.stream_helper import SPSHelper, write_ip, write_sps  # noqa: E402
    from src.utils.transforms import ycbcr420_to_444_np  # noqa: E402

    i_frame_net, p_frame_net = _init_models(
        model_i=model_i,
        model_p=model_p,
        force_zero_thres=force_zero_thres,
        device=device,
    )

    padding_r, padding_b = DMCI.get_padding_size(info.height, info.width, 16)
    use_two_entropy_coders = info.width * info.height > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

    output_buff = io.BytesIO()
    sps_helper = SPSHelper()
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    p_frame_net.set_curr_poc(0)
    last_qp = 0
    frame_index_map: list[int] = []
    encoded_frames = 0
    enc_idx = 0

    try:
        with torch.no_grad():
            for src_idx, frame in frame_iter:
                if max_frames >= 0 and encoded_frames >= max_frames:
                    break

                y, uv = bgr_to_yuv420_planes(frame)
                yuv = ycbcr420_to_444_np(y, uv)
                x = torch.from_numpy(yuv).to(device=device, dtype=torch.float32) / 255.0
                x = x.unsqueeze(0)
                if device.type == "cuda":
                    x = x.to(dtype=torch.float16)

                x_padded = replicate_pad(x, padding_b, padding_r)

                is_i = (enc_idx == 0) or (intra_period > 0 and enc_idx % intra_period == 0)
                if is_i:
                    curr_qp = qp_i
                    sps = {
                        "sps_id": -1,
                        "height": info.height,
                        "width": info.width,
                        "ec_part": 1 if use_two_entropy_coders else 0,
                        "use_ada_i": 0,
                    }
                    encoded = i_frame_net.compress(x_padded, qp_i)
                    p_frame_net.clear_dpb()
                    p_frame_net.add_ref_frame(None, encoded["x_hat"])
                    is_i_frame = True
                else:
                    fa_idx = index_map[enc_idx % 8]
                    if reset_interval > 0 and enc_idx % reset_interval == 1:
                        use_ada_i = 1
                        p_frame_net.prepare_feature_adaptor_i(last_qp)
                    else:
                        use_ada_i = 0
                    curr_qp = p_frame_net.shift_qp(qp_p, fa_idx)
                    sps = {
                        "sps_id": -1,
                        "height": info.height,
                        "width": info.width,
                        "ec_part": 1 if use_two_entropy_coders else 0,
                        "use_ada_i": use_ada_i,
                    }
                    encoded = p_frame_net.compress(x_padded, curr_qp)
                    last_qp = curr_qp
                    is_i_frame = False

                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                if sps_new:
                    write_sps(output_buff, sps)
                write_ip(output_buff, is_i_frame, sps_id, curr_qp, encoded["bit_stream"])

                frame_index_map.append(int(src_idx))
                encoded_frames += 1
                enc_idx += 1

        bitstream = output_buff.getvalue()
    finally:
        output_buff.close()

    meta = {
        "video_path": str(video_path),
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frames_total": info.frame_count,
        "frames_encoded": int(encoded_frames),
        "qp_i": qp_i,
        "qp_p": qp_p,
        "reset_interval": reset_interval,
        "intra_period": intra_period,
        "device": str(device),
        "use_kept_frames": bool(use_kept_frames),
        "frame_index_map": frame_index_map,
        "compressed_bytes": len(bitstream),
    }

    return {"bitstream_bytes": bitstream, "meta": meta}


def _init_models(model_i: str, model_p: str, force_zero_thres=None, device=None):
    from src.models.image_model import DMCI  # noqa: E402
    from src.models.video_model import DMC  # noqa: E402
    from src.utils.common import get_state_dict, set_torch_env  # noqa: E402

    set_torch_env()

    i_frame_net = DMCI()
    i_frame_net.load_state_dict(get_state_dict(str(model_i)))
    i_frame_net = i_frame_net.to(device).eval()
    i_frame_net.update(force_zero_thres)
    if device.type == "cuda":
        i_frame_net.half()

    p_frame_net = DMC()
    p_frame_net.load_state_dict(get_state_dict(str(model_p)))
    p_frame_net = p_frame_net.to(device).eval()
    p_frame_net.update(force_zero_thres)
    if device.type == "cuda":
        p_frame_net.half()

    return i_frame_net, p_frame_net


def encode_dcvc_frames_to_bytes(
    frames: Iterable[tuple[int, np.ndarray]],
    *,
    info: VideoInfo,
    cfg: dict[str, Any],
    video_path: str = "<rendered_frame_stream>",
) -> dict[str, Any]:
    return _encode_dcvc_core(
        frame_iter=frames,
        info=info,
        cfg=cfg,
        video_path=str(video_path),
        use_kept_frames=True,
    )
