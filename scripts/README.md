# Utility scripts

## download_models.py

Fetches the weights and sample data in `models/models.manifest.json`, verifying each against its SHA-256 before and after download.

```bash
python scripts/download_models.py --list              # groups and their files
python scripts/download_models.py --group compress    # edge side only
python scripts/download_models.py --group decompress restore
python scripts/download_models.py                     # everything (~2.5 GB)
python scripts/download_models.py --force             # re-download
```

Groups: `compress`, `decompress`, `restore`, `int16`, `data`.

Set `FAUNACODEC_ASSET_BASE` to download from a mirror instead of the GitHub release.

## sanity_check.py

End-to-end check on a synthetic clip with the codec mocked, so it needs neither weights nor a GPU. Use it to confirm an installation before downloading 2.5 GB.

```bash
python scripts/sanity_check.py
```

## eval_metrics.py

PSNR, SSIM, and LPIPS between a reconstruction and its source, per frame and aggregate.

```bash
python scripts/eval_metrics.py --pred outputs/pipeline/bird1_decompressed.mp4 \
                               --gt   data/bird1.mp4 \
                               --out-csv results/metrics.csv
```

Needs the `metrics` extra: `pip install -e ".[metrics]"`.
