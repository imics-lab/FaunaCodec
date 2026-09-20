"""Reading and writing video files."""
from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# mp4v first: it is the one FourCC every OpenCV build can write without an
# external H.264 encoder. The rest are tried only if the build supports them.
_CODEC_FALLBACKS = ["mp4v", "X264", "avc1", "XVID"]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def probe_video(path: str | Path) -> VideoInfo:
    """Read width, height, fps, and frame count without decoding the video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) for every frame in the video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_video(path: str | Path) -> tuple[list[np.ndarray], float]:
    """Read every frame into memory. Returns (frames, fps)."""
    info = probe_video(path)
    return [frame for _, frame in iter_frames(path)], info.fps


def resize_video(src: str | Path, dst: str | Path, width: int, height: int) -> Path:
    """Rewrite a video at an exact resolution, preserving its frame rate."""
    info = probe_video(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (int(width), int(height))
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {dst} at {width}x{height}")
    try:
        for _, frame in iter_frames(src):
            writer.write(cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA))
    finally:
        writer.release()
    return dst.resolve()


def write_video(
    frames: Iterable[np.ndarray],
    path: str | Path,
    fps: float,
    codec: str | None = None,
) -> Path:
    """Write BGR frames to an MP4, trying each fallback FourCC until one opens."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [codec] if codec else _CODEC_FALLBACKS

    writer: cv2.VideoWriter | None = None
    try:
        for frame in frames:
            if writer is None:
                height, width = frame.shape[:2]
                for name in candidates:
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*name), fps, (width, height)
                    )
                    if writer.isOpened():
                        break
                    writer.release()
                    writer = None
                if writer is None:
                    raise RuntimeError(
                        f"VideoWriter failed to open {path} with any of {candidates}, "
                        f"fps={fps}, size={width}x{height}"
                    )
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    if not path.exists():
        raise RuntimeError(f"VideoWriter produced no output file: {path}")
    return path.resolve()


def write_video_lossless(frames: Iterable[np.ndarray], path: str | Path, fps: float) -> Path:
    """Write BGR frames as FFV1 in an MKV container, for intermediates that must not lose quality."""
    frames = list(frames)
    if not frames:
        raise ValueError("write_video_lossless requires at least one frame")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{width}x{height}", "-r", str(max(1e-6, fps)), "-i", "-",
            "-an", "-c:v", "ffv1", "-level", "3", str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        stderr = proc.stderr.read().decode("utf-8", "replace").strip() if proc.stderr else ""
        raise RuntimeError(f"FFV1 encode failed (exit {proc.returncode}): {stderr}")
    return path.resolve()
