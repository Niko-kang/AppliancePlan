#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
闭环测试脚本 - 单机八卡单模型模式
用法示例：
python test_close_loop.py \
  --model_name your_model \
  --data_path /path/to/json_or_jsonl_dir \
  --root_path  \
  --result_dir /path/to/save
"""

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
import logging
import time
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# ---------------- 日志 ----------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("close_loop_eval")

# ---------------- GPU 配置（单机 8 卡）----------------
gpu_list = ["0", "1", "2", "3", "4", "5", "6", "7"]
#gpu_list = ["0"]

root = ""


def process_images(img_paths):
    """处理图像并返回处理后的图像列表和尺寸信息"""
    images = []
    image_sizes = []
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            image_sizes.append(img.size)
        except Exception as e:
            logger.error(f"Error loading image {p}: {str(e)}")
            placeholder = Image.new("RGB", (224, 224), (255, 255, 255))
            images.append(placeholder)
            image_sizes.append(placeholder.size)
    return images, image_sizes


def extract_file_info(filename):
    """从文件名提取信息"""
    # 例如: 002_breadmachine_ch_plan.json -> ("002", "breadmachine", "ch")
    parts = filename.replace(".json", "").split("_")
    if len(parts) >= 3:
        code = parts[0]
        appliance_type = parts[1]
        language = parts[2] if len(parts) > 2 else "en"
        return code, appliance_type, language
    return filename, "unknown", "en"


def parse_plan(plan_text: str):
    """解析 plan 为 (action, params[]) 列表"""
    pattern = r"([A-Za-z]+)\(([^)]*)\)"
    actions = []
    for name, params in re.findall(pattern, plan_text):
        params = [p.strip() for p in params.split(",") if p.strip()]
        actions.append((name.strip(), params))
    return actions


def compare_action_params(pred_action, gt_action):
    pred_name = pred_action[0].strip().lower()
    gt_name = gt_action[0].strip().lower()
    if pred_name != gt_name:
        return False

    pred_params = [re.sub(r"\s+", "", p.strip()).lower() for p in pred_action[1]]
    gt_params = [re.sub(r"\s+", "", p.strip()).lower() for p in gt_action[1]]

    if len(pred_params) != len(gt_params):
        return False

    if len(gt_params) in (0, 1):
        return pred_params == gt_params

    if len(gt_params) == 2:
        return pred_params == gt_params

    if len(gt_params) == 3:
        return pred_params[0] == gt_params[0] and pred_params[2] == gt_params[2]

    return pred_params == gt_params


def exact_match_by_actions(pred_actions, gt_actions):
    if len(pred_actions) != len(gt_actions):
        return False

    for pred_action, gt_action in zip(pred_actions, gt_actions):
        if not compare_action_params(pred_action, gt_action):
            return False

    return True


def step_by_step_match(pred_actions, gt_actions):
    if not pred_actions or not gt_actions:
        return 0.0

    min_len = min(len(pred_actions), len(gt_actions))
    correct_count = 0

    for i in range(min_len):
        if compare_action_params(pred_actions[i], gt_actions[i]):
            correct_count += 1
        else:
            break

    return correct_count / len(gt_actions) if gt_actions else 0.0


def score_plan(predicted: str, ground_truth: str) -> dict:
    """综合评分函数：返回两种测评方式的结果"""
    if not predicted or not ground_truth:
        return {"exact_match": 0.0, "step_match": 0.0}

    pred_actions = parse_plan(predicted)
    gt_actions = parse_plan(ground_truth)

    if not pred_actions or not gt_actions:
        return {"exact_match": 0.0, "step_match": 0.0}

    # 1. 完全匹配：基于原子动作和参数的匹配，忽略空格等因素
    exact_match = 1.0 if exact_match_by_actions(pred_actions, gt_actions) else 0.0

    # 2. 逐步匹配
    step_match = step_by_step_match(pred_actions, gt_actions)

    return {"exact_match": exact_match, "step_match": step_match}


def test_totally_same(input_jsonl):
    """计算两种测评方式的准确率并写出到同名 .txt"""
    input_jsonl = Path(input_jsonl)
    total = 0
    exact_correct = 0
    step_sum = 0.0
    per_item_scores = []
    non_exact_cases = []

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"[WARN] 第 {line_no} 行 JSON 解析失败：{e}")
                continue

            total += 1
            pred = data.get("answer", "").strip()
            gt = data.get("Ground Truth", "").strip().lower()

            scores = score_plan(pred, gt)

            exact = 1.0 if scores.get("exact_match", 0.0) == 1.0 else 0.0
            step = float(scores.get("step_match", 0.0))

            exact_correct += exact
            step_sum += step

            per_item_scores.append(
                {"id": data.get("id"), "exact": exact, "step": round(step, 6)}
            )

            if exact == 0.0:
                non_exact_cases.append(
                    {
                        "id": data.get("id"),
                        "step": round(step, 6),
                        "predicted": pred,
                        "ground_truth": gt,
                    }
                )

    exact_acc = exact_correct / total if total else 0.0
    step_acc = step_sum / total if total else 0.0

    logger.info(f"Total: {total}")
    logger.info(f"Exact Match Accuracy: {exact_acc:.4f}")
    logger.info(f"Step Match Accuracy:  {step_acc:.4f}")

    txt_path = input_jsonl.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as ft:
        ft.write(f"Total: {total}\n")
        ft.write(f"Exact Match Accuracy: {exact_acc:.4f}\n")
        ft.write(f"Step Match Accuracy:  {step_acc:.4f}\n\n")

        ft.write("Per-item scores:\n")
        for item in per_item_scores:
            ft.write(json.dumps(item, ensure_ascii=False) + "\n")

        ft.write("\nNon-exact cases (gt vs pred):\n")
        for item in non_exact_cases:
            ft.write(f"ID: {item.get('id', 'N/A')}\n")
            ft.write(f"Step Match: {item.get('step', 0):.6f}\n")
            ft.write(f"Predicted-----:\n{item.get('predicted', '')}\n")
            ft.write(f"\nGround Truth--:\n{item.get('ground_truth', '')}\n")
            ft.write("-" * 80 + "\n")

    logger.info(f"All results saved to: {txt_path.absolute()}")


def run_close_loop_inference(
    llm,
    processor,
    test_data,
    output_file,
    max_new_tokens=512,
):
    """运行闭环推理"""
    results = []

    for sample in tqdm(test_data, desc="Processing samples"):
        try:
            # 获取图像路径
            img_paths = sample.get("image", [])
            print(img_paths)
            if isinstance(img_paths, str):
                img_paths = [img_paths]
            elif not isinstance(img_paths, list):
                logger.error(
                    f"Error processing sample {sample.get('id', 'unknown')}: invalid image field type {type(img_paths)}"
                )
                continue
            if not img_paths:
                logger.warning(
                    f"No images found for sample {sample.get('id', 'unknown')}"
                )
                continue

            # 处理图像
            images, image_sizes = process_images(img_paths)

            # 构建消息
            conversations = sample.get("conversations", [])
            user_message = None
            assistant_message = None

            for conv in conversations:
                if conv.get("from") == "human":
                    user_message = conv.get("value", "")
                elif conv.get("from") == "gpt":
                    assistant_message = conv.get("value", "")

            if not user_message:
                logger.warning(
                    f"No user message found for sample {sample.get('id', 'unknown')}"
                )
                continue

            # 构建输入消息
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_message}]
                    + [{"image": img} for img in images],
                }
            ]

            # 生成模型输入（与test.py/test_keypage.py一致）
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            multi_modal_data = {"image": [img for img in images]}

            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=0.25,
                max_tokens=max_new_tokens,
                stop=None,
            )

            # 运行推理
            llm_inputs = {"prompt": prompt, "multi_modal_data": multi_modal_data}
            outputs = llm.generate([llm_inputs], sampling_params=sampling_params)

            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text.strip().lower()

                # 如果模型预测多步，只截取第一步
                if "\n" in generated_text:
                    generated_text = generated_text.split("\n")[0].strip()

                pred_actions = parse_plan(generated_text)
                if pred_actions:
                    first_action = pred_actions[0]
                    first_step = f"{first_action[0]}({', '.join(first_action[1])})"
                    generated_text = first_step

                # 构建结果
                result = {
                    "id": sample.get("id", ""),
                    "file_info": sample.get("file_info", {}),
                    "image": img_paths,
                    "question": user_message,
                    "answer": generated_text,
                    "Ground Truth": (assistant_message or "").strip().lower(),
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)

                # 保存到文件（实时写入）
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            else:
                logger.warning(
                    f"No output generated for sample {sample.get('id', 'unknown')}"
                )

        except Exception as e:
            logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
            continue

    return results


def process_file_list_single_instance(
    model_name,
    json_file_list,
    gpu_id,
    root_path,
    custom_result_dir,
):
    """在独立进程中处理一批JSON文件，固定到指定GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # 路径配置
        if os.path.isabs(model_name):
            model_path = model_name
        else:
            model_path = os.path.join(root_path, "ft/scripts/Model_ftd", model_name)

        save_base = (
            custom_result_dir
            if custom_result_dir
            else os.path.join(root_path, "manual_data/Train", model_name, "result/")
        )
        os.makedirs(save_base, exist_ok=True)

        logger.info(
            f"[GPU {gpu_id}] Loading model ONCE for {len(json_file_list)} files: {model_name}"
        )

        processor = AutoProcessor.from_pretrained(model_path)
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            limit_mm_per_prompt={"image": 25},
        )
        logger.info(f"[GPU {gpu_id}] Model loaded.")

        all_results = []
        data_dir = None

        for json_file in json_file_list:
            try:
                code, appliance_type, language = extract_file_info(json_file.name)

                with open(json_file, "r", encoding="utf-8") as f:
                    test_data = json.load(f)

                # 为每个样本添加文件信息
                for sample in test_data:
                    sample["file_info"] = {
                        "filename": json_file.name,
                        "code": code,
                        "appliance_type": appliance_type,
                        "language": language,
                    }

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(
                    save_base,
                    f"close_loop_{json_file.stem}_inference_{timestamp}.jsonl",
                )

                logger.info(f"[GPU {gpu_id}] Processing file: {json_file.name}")
                results = run_close_loop_inference(
                    llm, processor, test_data, output_file
                )
                all_results.extend(results)

                # 评分与保存检测结果
                test_totally_same(output_file)

                data_dir = json_file.parent

            except Exception as e:
                logger.exception(f"[GPU {gpu_id}] Error processing {json_file}: {e}")

        logger.info(f"✅ [GPU {gpu_id}] Finished {len(json_file_list)} files.")
        return (model_name, all_results, data_dir)

    except Exception as e:
        logger.exception(
            f"❌ [GPU {gpu_id}] Error in process_file_list_single_instance: {e}"
        )
        return (model_name, [], None)


def main():
    """主函数 - 单机八卡、每卡单模型"""
    global root

    parser = argparse.ArgumentParser(
        description="Close Loop testing system - Single Node 8 GPU, 1 instance per GPU"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name to test"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Directory that contains *.json test files",
    )
    parser.add_argument("--root_path", type=str, default=root, help="Root path")
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Result directory for saving evaluation results",
    )
    args = parser.parse_args()

    root = args.root_path

    # 收集所有JSON文件
    data_path = Path(args.data_path)
    json_files = list(data_path.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON files found in {args.data_path}")
        return

    logger.info(f"Found {len(json_files)} JSON files to process")

    # 将文件分配给GPU
    files_per_gpu = len(json_files) // len(gpu_list)
    remainder = len(json_files) % len(gpu_list)

    file_assignments = []
    start_idx = 0
    for i, gpu_id in enumerate(gpu_list):
        num_files = files_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + num_files
        if start_idx < len(json_files):
            file_assignments.append((gpu_id, json_files[start_idx:end_idx]))
        start_idx = end_idx

    logger.info(f"Assigned {len(file_assignments)} GPU tasks")

    # 并行处理
    start_time = time.time()
    all_results_by_model = {}
    completed_count = 0
    total_tasks = len(file_assignments)

    with ProcessPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures = {
            executor.submit(
                process_file_list_single_instance,
                args.model_name,
                file_list,
                gpu_id,
                args.root_path,
                args.result_dir,
            ): (gpu_id, file_list)
            for gpu_id, file_list in file_assignments
        }

        for fut in as_completed(futures):
            gpu_id, file_list = futures[fut]
            completed_count += 1
            try:
                ret = fut.result()
                if ret and len(ret) == 3:
                    returned_model_name, returned_results, data_dir = ret
                    if returned_model_name not in all_results_by_model:
                        all_results_by_model[returned_model_name] = {
                            "results": [],
                            "data_dir": data_dir,
                        }
                    all_results_by_model[returned_model_name]["results"].extend(
                        returned_results
                    )
                    if data_dir:
                        all_results_by_model[returned_model_name]["data_dir"] = data_dir
                logger.info(
                    f"✅ [{completed_count}/{total_tasks}] GPU {gpu_id} finished."
                )
            except Exception as e:
                logger.error(
                    f"❌ [{completed_count}/{total_tasks}] GPU {gpu_id} failed: {e}"
                )

    duration = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(
        f"All tasks completed in {duration:.2f} seconds ({duration/60:.2f} minutes)"
    )
    logger.info(f"Total files processed: {len(json_files)}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
