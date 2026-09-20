# Third-party software

FaunaCodec vendors the external implementations that are required at runtime or carry
local compatibility patches. Keeping them here makes a clone reproducible without
Git submodules. Model weights are distributed separately through the release manifest
in `models/models.manifest.json`.

| Directory | Upstream | License | Status | Used by |
|---|---|---|---|---|
| `dcvc/` | [microsoft/DCVC](https://github.com/microsoft/DCVC) | MIT | Vendored snapshot; exact upstream revision was not recorded | Learned image and video compression |
| `amt/` | [MCG-NKU/AMT](https://github.com/MCG-NKU/AMT) | CC BY-NC 4.0 | Reduced vendored snapshot; exact upstream revision was not recorded | Frame interpolation during decompression |
| `osediff/` | [cswry/OSEDiff](https://github.com/cswry/OSEDiff) | Apache-2.0 | Reduced snapshot with local compatibility patches; exact upstream revision was not recorded | Optional one-step diffusion upscaling |
| `dcvc_int16/` | Derived from [microsoft/DCVC](https://github.com/microsoft/DCVC) | MIT | FaunaCodec-maintained derivative with project-specific reference models, CUDA kernels, and rANS bindings | Deterministic cross-architecture codec runtime |

The component directories retain their own license, notice, and attribution files.
`THIRD_PARTY_LICENSES.md` summarizes how their terms affect the combined distribution.
