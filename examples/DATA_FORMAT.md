# Data format (UseAppliance-style)

Training samples follow the multimodal ShareGPT layout used for **AppliancePlan**
fine-tuning (see the ACM MM ’26 paper / UseAppliance).

All task JSON files are a **list of samples**. Each sample is a dict with at least:

- `image`: list of image paths (strings)
- `conversations`: ShareGPT-style turns with `from` = `human` | `gpt` and `value` text  
  Use `<image>` placeholders in the human turn; count must match `len(image)`.

Optional fields seen in training data: `id`, `task`, `item_info`, …

## Image paths

Paths in JSON are joined with `--image_folder` / `IMAGE_PATH` (training) or resolved via `--root_path` (eval), depending on the script.

For the bundled demo, paths are **repo-relative**, e.g. `examples/images/demo_page.png`.  
Set:

```bash
export IMAGE_PATH=/path/to/Data_Flywheel_OpenSource
# or absolute paths inside your own JSON
```

## Task flavors (examples)

| File | Task |
|------|------|
| `examples/exp_demo/iteration_0/Train_data/demo_plan.json` | open-loop plan |
| `examples/sample_bbox.json` | part / bbox grounding |
| `examples/sample_keypage.json` | keypage yes/no |
| `examples/sample_closeloop.json` | close-loop state QA |

## Experiment layout expected by `train_flywheel.sh`

```text
${WORK_ROOT}/${EXP_NAME}/
  iteration_0/
    Train_data/*.json
    Test_data/*.json
    plan_test/ bbox_test/ keypage_test/ close_loop_test/   # optional eval sets
  real_*_test_data/                                        # optional held-out
```

Copy `examples/exp_demo` as a starting point:

```bash
export WORK_ROOT=/tmp/Flywheel_Train
mkdir -p "$WORK_ROOT"
cp -r examples/exp_demo "$WORK_ROOT/demo-exp"
```

## Models / full datasets

This repository ships **code + tiny placeholders only**.  
Base models (e.g. Qwen2.5-VL) and full manuals/images must be obtained separately under their own licenses.
