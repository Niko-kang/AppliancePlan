# AppliancePlan / UseAppliance (Data Flywheel)

Official code for the ACM MM ’26 paper:

> **Scaling Manual-Grounded Appliance Manipulation with Data Synthesis and Unified Planning**  
> Yuxing Long\*, Lei Kang\*, Ziyan Yu, Yuzheng Gao, Bin Cheng, Jiyao Zhang, Xiaoqi Li, Haolin Yang, Dongjiang Li, Hui Shen, Hao Dong†  
> [ACM DL](https://doi.org/10.1145/3767308.3836196) · PDF: [`docs/paper/AppliancePlan_MM26.pdf`](docs/paper/AppliancePlan_MM26.pdf)

This repository releases the **training & evaluation stack** used to build **AppliancePlan** on the **UseAppliance** data synthesized by **MAGE** (manual-grounded appliance data generation with Hierarchical Appliance Graphs).

| Paper concept | What we release |
|---------------|-----------------|
| **MAGE** data pipeline outputs | Training JSON + images (HF Dataset) |
| **AppliancePlan** | Fine-tuned VLM weights (HF Model) + train/eval code |
| **RealAppliance-Bench**-style tasks | Eval scripts: plan / bbox / keypage / close-loop |

\* Equal contribution. † Corresponding author (`hao.dong@pku.edu.cn`).

License: [Apache-2.0](LICENSE) (code). See the paper for CC BY 4.0 terms on the article itself.

## Repository layout

```text
AppliancePlan/     # this repo → GitHub
  src/                        # SFT / DPO / GRPO training
  eval/                       # plan, bbox, keypage, close-loop evaluation
  scripts/train_flywheel.sh   # train → eval loop (optional multi-round)
  optional/data_enhancement/  # between-round train-set expansion (off by default)
  docs/                       # environment + paper PDF
  examples/                   # tiny format demos

../AppliancePlan-Model/    # → Hugging Face (model)
../UseAppliance/  # → Hugging Face (dataset: Train_data + images)
```

## Tasks (aligned with the paper)

1. **Part grounding (bbox)** — localize appliance parts from manuals / observations  
2. **Open-loop planning** — long-horizon atomic-action plans grounded on manuals  
3. **Key-page / auxiliary** — keypage & related judgments used in training  
4. **Closed-loop adjustment** — revise plans under disturbances / state feedback  

Default flywheel setting for release: `MAX_ITERATIONS=1`, `ENABLE_DATA_ENHANCEMENT=0`  
(multi-round + pool sampling remain available for research reproduction).

## Quick start

### 1) Environment

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

```bash
conda create -n flywheel-train python=3.11 -y && conda activate flywheel-train
pip install -r requirements-train.txt   # install a CUDA-matched torch wheel first if needed

conda create -n flywheel-infer python=3.11 -y && conda activate flywheel-infer
pip install -r requirements-infer.txt
```

### 2) Model & training data

```bash
export MODEL_PATH=/home/jd/下载/开源数据/AppliancePlan-Model
export IMAGE_PATH=/home/jd/下载/开源数据/UseAppliance
# annotations: $IMAGE_PATH/Train_data
# images:      $IMAGE_PATH/images/...  (JSON paths are relative to IMAGE_PATH)
```

After you upload to Hugging Face, replace local paths with `huggingface-cli download ...`.

### 3) Train → eval

Edit `CONFIGURATIONS` in `scripts/train_flywheel.sh`, then:

```bash
export MODEL_PATH=/home/jd/下载/开源数据/AppliancePlan-Model   # or a base Qwen2.5-VL
export IMAGE_PATH=/home/jd/下载/开源数据/UseAppliance
export ROOT_PATH=$IMAGE_PATH
export WORK_ROOT=/tmp/Flywheel_Train
export MAX_ITERATIONS=1
export ENABLE_DATA_ENHANCEMENT=0

bash scripts/train_flywheel.sh
```

Tune `GLOBAL_BATCH_SIZE` / `NUM_DEVICES` inside the script for your GPUs.

### 4) Eval only

```bash
bash scripts/run_all_evals_single_node.sh \
  --model_path "$MODEL_PATH" \
  --root_path  "$IMAGE_PATH" \
  --out_dir    /path/to/eval_out \
  --plan_data  /path/to/plan_test \
  --bbox_data  /path/to/bbox_test \
  --keypage_data /path/to/keypage_test \
  --close_loop_data /path/to/close_loop_test
```

### 5) Merge JSON (no GPU)

```bash
python tools/merge_train_data.py \
  --input_dir "$IMAGE_PATH/Train_data" \
  --output_file /tmp/useappliance_train.json
```

## Citation

If you use this code, model, or data, please cite:

```bibtex
@inproceedings{long2026applianceplan,
  title     = {Scaling Manual-Grounded Appliance Manipulation with Data Synthesis and Unified Planning},
  author    = {Long, Yuxing and Kang, Lei and Yu, Ziyan and Gao, Yuzheng and Cheng, Bin
               and Zhang, Jiyao and Li, Xiaoqi and Yang, Haolin and Li, Dongjiang
               and Shen, Hui and Dong, Hao},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  address   = {Rio de Janeiro, Brazil},
  publisher = {ACM},
  doi       = {10.1145/3767308.3836196},
  url       = {https://doi.org/10.1145/3767308.3836196}
}
```

## Acknowledgements

Built on Qwen2.5-VL and the broader open-source VLM / DeepSpeed / vLLM ecosystem.  
RealAppliance-Bench and related digital assets are described in the paper and prior work.
