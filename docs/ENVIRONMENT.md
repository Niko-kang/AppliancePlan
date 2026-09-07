# Environment setup

## Hardware

- NVIDIA GPU + recent CUDA driver recommended
- Training script defaults assume multi-GPU DeepSpeed; adjust `GLOBAL_BATCH_SIZE` / `BATCH_PER_DEVICE` / `NUM_DEVICES` in `scripts/train_flywheel.sh` for your machine

## Python envs (recommended split)

Training and vLLM eval often use different stacks. Two conda envs is fine:

```bash
# --- train ---
conda create -n flywheel-train python=3.11 -y
conda activate flywheel-train
pip install -r requirements-train.txt

# --- infer / eval ---
conda create -n flywheel-infer python=3.11 -y
conda activate flywheel-infer
pip install -r requirements-infer.txt
```

Or install everything from `requirements.txt` if your stack allows it.

## Important env vars

| Variable | Meaning |
|----------|---------|
| `MODEL_PATH` | HF / local Qwen2.5-VL path or fine-tuned checkpoint |
| `ROOT_PATH` / `IMAGE_PATH` | Workspace / image root |
| `WORK_ROOT` | Experiment outputs (`Flywheel_Train/...`) |
| `MAX_ITERATIONS` | Train→eval rounds (default `1`) |
| `ENABLE_DATA_ENHANCEMENT` | Between-round pool sampling (default `0`) |
| `PET_NODE_RANK` / `PET_NNODES` | Multi-node rank (default `0` / `1`) |

## Smoke checks (no GPU model needed)

```bash
python tools/merge_train_data.py \
  --input_dir examples/exp_demo/iteration_0/Train_data \
  --output_file /tmp/demo_train.json
```

## What is not shipped

- Full training/eval datasets and images  
- Model weights  
- Internal cluster sshd bootstrap (intentionally omitted for security)
