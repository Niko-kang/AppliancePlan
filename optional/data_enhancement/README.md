# Optional: between-round data enhancement

These scripts are **not** part of the default open-source pipeline.

They run only when:

1. `MAX_ITERATIONS > 1` (or you still prepare `iteration_{n+1}`), and  
2. `ENABLE_DATA_ENHANCEMENT=1` in `scripts/train_flywheel.sh`.

## Behavior

After a round’s eval, each script reads the merged metric summary:

| Script | Metric | Default skip threshold | Action if below threshold |
|--------|--------|------------------------|---------------------------|
| `data_enhancement.py` | Overall Exact Match Average | 0.7 | Sample ~50% from `data_enhancement/` |
| `data_enhancement_bbox.py` | Average IoU | 0.5 | Sample from `data_enhancement_bbox/` |
| `data_enhancement_keypage.py` | Recall & F1 | both ≥ 0.7 | Sample from `data_enhancement_keypage/` |
| `data_enhancement_close_loop.py` | Exact Match Accuracy | 0.7 | Sample from `data_enhancement_close_loop/` |

Samples are written into the **next** iteration’s `Train_data/` as `enhanced_*_iter_{n}.json`.

## Why it is optional

The public release targets a clean **train → eval** flow (`MAX_ITERATIONS=1`).  
Auto-growing the training set between rounds is a research flywheel detail; keeping the code here preserves reproducibility without forcing it on every user.
