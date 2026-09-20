# Training-pair generation

`prepare_restoration.py` builds the dataset RestoreUNet trains on: pairs of clean source frames and the same frames after a full compress/decompress round trip. The degraded half is produced by the real pipeline, so the model is trained on exactly the artifacts it will see at inference.

```bash
python data_prep/prepare_restoration.py \
    --videos  /path/to/raw/clips \
    --output  data/restoration_pairs \
    --resolution 640x360 \
    --max-frames-per-video 150
```

Requires the compression and decompression weights (`python scripts/download_models.py --group compress decompress`).

Output layout, one subdirectory per source clip:

```
data/restoration_pairs/
  original/<clip>/000000.png ...    clean frames
  degraded/<clip>/000000.png ...    the same frames after the round trip
```

Full frames are written rather than crops: the training Dataset applies one random spatial crop across the whole temporal window, which requires the frames to still be spatially aligned.

Already-processed clips are skipped, so the script is resumable.

Then train with [`training/train_restoration.py`](../training/README.md).
