# dcvc_int16: deterministic int16 DCVC runtime

Standalone encoder/decoder pair for DCVC-RT with integer-quantized (int16) inference.

Floating-point DCVC is not bit-exact across machines: a stream encoded on one and decoded on another reconstructs into noise, with no error raised anywhere. The top-level [README](../../README.md#the-cross-architecture-trap) shows what that looks like between an ARM edge device and an x86 server. Integer-quantized inference removes the divergence, so it is the operational codec for the edge/server split: the edge device encodes, the server decodes, and both agree on every byte.

Derived from the Microsoft DCVC project (MIT license, see NOTICE); the int16 reference models, CUDA kernels, and rANS bindings under `src/` are this project's additions.

## Setup

```bash
cd third_party/dcvc_int16
python bootstrap_runtime.py     # installs deps and builds the C++/CUDA extensions
```

`build_int16_cuda.py` rebuilds just the CUDA kernels if needed.

Fetch the weight bundle into `models/` first:

```bash
python ../../scripts/download_models.py --group int16
```

## Usage

```bash
# Encode: mp4 -> .bin (writes the bitstream plus a JSON metrics/metadata sidecar)
python encode_mp4_to_bin.py --input_mp4 /path/to/clip.mp4 \
    --bundle_path ../../models/int16_bundle_v1.0.0.pt --output_dir outputs

# Decode: .bin -> mp4
python decode_bin_to_mp4.py --input_bin outputs/clip.bin \
    --bundle_path ../../models/int16_bundle_v1.0.0.pt --output_dir outputs
```

The encoder records environment, model-bundle, and git metadata in the sidecar so two runs can be audited for equivalence; matching bundle and bitstream hashes imply bit-exact reconstruction on any supported GPU.
