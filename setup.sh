#!/usr/bin/env bash
# Install FaunaCodec into a virtual environment and build the DCVC entropy-coder extension.
#
#   bash setup.sh                       # venv in ./.venv, auto-detect CUDA
#   bash setup.sh --cuda 12.4           # pin the PyTorch CUDA wheel
#   bash setup.sh --venv ~/envs/fauna   # put the venv somewhere else
#   bash setup.sh --python python3.11   # interpreter to build the venv from
#   bash setup.sh --no-venv             # install into the active environment
#   bash setup.sh --extras upscale      # also install the diffusion upscaler stack
set -euo pipefail

CUDA_VERSION=""
BASE_PYTHON="python3"
USE_VENV=true
EXTRAS="metrics"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT}/.venv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)    CUDA_VERSION="$2"; shift 2 ;;
        --venv)    VENV_DIR="$2"; shift 2 ;;
        --python)  BASE_PYTHON="$2"; shift 2 ;;
        --extras)  EXTRAS="$2"; shift 2 ;;
        --no-venv) USE_VENV=false; shift ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Pick the PyTorch wheel index ─────────────────────────────────────────────
ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" ]]; then
    # aarch64 (Jetson, GB10) has no wheels on download.pytorch.org. Those platforms
    # ship PyTorch through NVIDIA's own channel, so leave whatever is installed alone.
    echo "Detected $ARCH: skipping the PyTorch wheel index."
    echo "Install PyTorch from NVIDIA's channel for this platform before continuing."
    TORCH_INDEX=""
else
    if [[ -z "$CUDA_VERSION" ]]; then
        if command -v nvcc &>/dev/null; then
            CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        elif command -v nvidia-smi &>/dev/null; then
            CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
        else
            echo "No CUDA toolkit found; installing CPU-only PyTorch."
            CUDA_VERSION="cpu"
        fi
    fi
    case "$CUDA_VERSION" in
        cpu)       TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
        12.6|12.7|12.8|13.*) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
        12.4|12.5) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        *)         TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    esac
fi

echo "============================================================"
echo " FaunaCodec setup"
echo "   architecture: $ARCH"
echo "   CUDA:         ${CUDA_VERSION:-n/a}"
echo "   torch index:  ${TORCH_INDEX:-<platform default>}"
echo "   environment:  $($USE_VENV && echo "$VENV_DIR" || echo '<active env>')"
echo "   extras:       $EXTRAS"
echo "============================================================"

if ! command -v ffmpeg &>/dev/null; then
    echo "Warning: ffmpeg is not on PATH. The H.264/HEVC/AV1 backends and the lossless"
    echo "         intermediates need it: 'sudo apt install ffmpeg' (Debian/Ubuntu),"
    echo "         'brew install ffmpeg' (macOS)."
fi

# ── 1. Environment ───────────────────────────────────────────────────────────
if $USE_VENV; then
    command -v "$BASE_PYTHON" &>/dev/null || { echo "$BASE_PYTHON not found; pass --python." >&2; exit 1; }
    "$BASE_PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || {
        echo "FaunaCodec needs Python 3.10 or newer." >&2
        echo "Point setup.sh at one, e.g. --python python3.11." >&2
        exit 1
    }
    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        echo "[1/4] Reusing the virtual environment in ${VENV_DIR}."
    else
        echo "[1/4] Creating a virtual environment in ${VENV_DIR} ..."
        "$BASE_PYTHON" -m venv "$VENV_DIR" || {
            echo "Creating the virtual environment failed; see the error above." >&2
            echo "On Debian/Ubuntu the venv module is packaged separately:" >&2
            echo "  sudo apt install python3-venv" >&2
            exit 1
        }
    fi
    PYTHON="${VENV_DIR}/bin/python"
else
    echo "[1/4] Using the active environment."
    PYTHON="$(command -v python3 || command -v python)"
fi

# ── 2. PyTorch, then FaunaCodec ──────────────────────────────────────────────
echo "[2/4] Installing dependencies ..."
"$PYTHON" -m pip install --upgrade pip
if [[ -n "$TORCH_INDEX" ]]; then
    "$PYTHON" -m pip install --index-url "$TORCH_INDEX" torch torchvision
fi
if [[ -n "$EXTRAS" ]]; then
    "$PYTHON" -m pip install -e "${ROOT}[${EXTRAS}]"
else
    "$PYTHON" -m pip install -e "$ROOT"
fi

# ── 3. DCVC entropy-coder extension ──────────────────────────────────────────
echo "[3/4] Building the DCVC C++ entropy coder ..."
DCVC_CPP="${ROOT}/third_party/dcvc/src/cpp"
if [[ ! -d "$DCVC_CPP" ]]; then
    echo "DCVC is missing from ${ROOT}/third_party/dcvc." >&2
    echo "It is vendored in this repository; re-clone FaunaCodec, or fetch it with:" >&2
    echo "  git clone https://github.com/microsoft/DCVC ${ROOT}/third_party/dcvc" >&2
    exit 1
fi
"$PYTHON" -m pip install "pybind11>=2.11" "setuptools>=69"
( cd "$DCVC_CPP" && "$PYTHON" -m pip install --no-build-isolation . )

# ── 4. Verify ────────────────────────────────────────────────────────────────
echo "[4/4] Verifying ..."
"$PYTHON" - <<'PYEOF'
import sys

failures = []
for label, probe in [
    ("torch", lambda: __import__("torch")),
    ("opencv", lambda: __import__("cv2")),
    ("ultralytics", lambda: __import__("ultralytics")),
    ("faunacodec", lambda: __import__("faunacodec")),
    ("MLCodec_extensions_cpp", lambda: __import__("MLCodec_extensions_cpp")),
]:
    try:
        module = probe()
        version = getattr(module, "__version__", "ok")
        print(f"  {label:<24} {version}")
    except Exception as exc:
        failures.append(f"{label}: {exc}")

try:
    import torch
    print(f"  {'CUDA available':<24} {torch.cuda.is_available()}")
except Exception:
    pass

if failures:
    print("\nFailed:")
    for failure in failures:
        print(f"  {failure}")
    sys.exit(1)
print("\nAll imports OK.")
PYEOF

cat <<EOF

Setup complete.
$($USE_VENV && echo "  source ${VENV_DIR}/bin/activate")
  python scripts/download_models.py --group compress decompress
  faunacodec-pipeline data/bird1.mp4 --stages compress decompress
EOF
