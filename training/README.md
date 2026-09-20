# Training RestoreUNet

The restoration stage is optional and a trained checkpoint ships with the release (`python scripts/download_models.py --group restore`). Train your own only if you are adapting the model to different footage or a different codec operating point.

## Data

`train_restoration.py` reads pairs of clean and round-tripped frames:

```
data/restoration_pairs/
  original/<clip>/000000.png ...
  degraded/<clip>/000000.png ...
```

Generate them with [`data_prep/prepare_restoration.py`](../data_prep/README.md). The degraded half must come from the same codec settings you will use at inference, since the model learns that codec's artifacts specifically.

## Run

```bash
python training/train_restoration.py \
    --data       data/restoration_pairs \
    --config     configs/restoration.yaml \
    --ckpt-dir   checkpoints/restore_l1 \
    --epochs     200 \
    --batch-size 8 \
    --save-every 10 \
    --sample-every 20
```

| Option | Default | Notes |
|---|---|---|
| `--lr` | 3e-4 | |
| `--ema-decay` | 0.995 | EMA of the model weights, used for sampling |
| `--vgg-weight` | from config (0) | See below |
| `--resume PATH` | | Resume from a checkpoint |

**Keep `--vgg-weight` at 0.** The loss is an L1 diffusion noise-prediction loss. Adding VGG perceptual loss teaches the model to synthesize plausible texture that does not match the ground truth, which degrades PSNR, SSIM, LPIPS, and VMAF together. Restoration is meant to remove artifacts faithfully; synthesizing detail is the upscaling stage's job, and it is kept separate and optional for exactly this reason.

## Checkpoints

Each is a dict with `model`, `ema`, `opt`, `epoch`, and `model_cfg`. `best.pt` points at the lowest validation loss.

The `restoration_T1.yaml` and `restoration_T5.yaml` configs vary the temporal attention window (1 and 5 frames against the default 3); each needs its own training run and its own checkpoint directory.

## Reading the logs

- `epoch N  loss=...  val_loss=...` — per-epoch summary
- `[sample] saved sample grid to samples/epochN.png` — degraded input, model output, ground truth, left to right
- `[ckpt] saved ...` — checkpoint written

If the output looks blurrier than the degraded input, the pairs are misaligned: check that the filenames under `original/` and `degraded/` correspond frame for frame.
