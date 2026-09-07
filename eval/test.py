#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单机八卡推理系统 - 单卡单模型模式

用法示例:
python test.py --model_name your_model_name --data_path /path/to/json_dir --result_dir /path/to/save

特性:
- 单机8 GPU并行推理
- 每张GPU运行一个模型实例
- 文件轮询分配给8张GPU
- 内存优化：限制 gpu_memory_utilization；启用 expandable_segments
- 自动生成双指标评估报告（Exact / Step）
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
from PIL import Image, ImageOps

# --------- 重要的内存/碎片化优化（尽早设置） ----------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------- 导入模型相关代码 ----------
import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info  # 你项目内的工具，如未用到可去掉引入

# ---------- 全局配置 ----------
root = ""
data_base = None  # 数据路径，通过命令行参数设置

# ---------- 模型配置 ----------
model_set = [
    # 默认模型列表，可以通过命令行参数覆盖
]

# ---------- GPU配置 ----------
gpu_list = ["0", "1", "2", "3", "4", "5", "6", "7"]  # 单机8张GPU

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ========================= 基础工具函数 =========================


def process_images(img_paths):
    """处理图像并返回处理后的图像列表和尺寸信息（保持原图尺寸）"""
    images = []
    image_sizes = []

    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            # 保持原图尺寸，不做max_size限制
            images.append(img)
            image_sizes.append(img.size)
        except Exception as e:
            logger.error(f"Error loading image {p}: {str(e)}")
            placeholder = Image.new("RGB", (224, 224), (255, 255, 255))
            images.append(placeholder)
            image_sizes.append(placeholder.size)

    return images, image_sizes


def extract_file_info(filename):
    """
    从文件名中提取信息：编码_家电类型_语言
    返回: (编码, 家电类型, 语言)
    """
    name_without_ext = Path(filename).stem
    parts = name_without_ext.split("_", 2)

    if len(parts) >= 3:
        code = parts[0]
        appliance_type = parts[1]
        language = parts[2]
    elif len(parts) == 2:
        code = parts[0]
        appliance_type = parts[1]
        language = "unknown"
    else:
        code = name_without_ext
        appliance_type = "unknown"
        language = "unknown"

    return code, appliance_type, language


# ========================= 推理与评估 =========================


def run_pl_inference(llm, processor, test_data, output_file, max_new_tokens=1024):
    """使用大模型进行plan推理"""
    try:
        eos_id = getattr(processor.tokenizer, "eos_token_id", None)
        stop_ids = [eos_id] if isinstance(eos_id, int) else None

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
        )

        results = []

        for sample in tqdm(test_data, desc="PL Inference"):
            # 获取问题和图片路径
            question = sample["conversations"][0]["value"]
            img_paths = sample.get("image", [])

            # 处理图像
            images, image_sizes = process_images(img_paths)

            # 构造messages
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}]
                    + [{"image": img} for img in images],
                }
            ]

            # 生成模型输入
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 准备多模态输入
            multi_modal_data = {"image": [img for img in images]}

            # 如果你确实需要可保留；若未使用可注释掉，避免多余开销
            # _image_inputs, _video_inputs, _video_kwargs = process_vision_info(
            #     messages, return_video_kwargs=True
            # )

            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": multi_modal_data,
            }

            # 生成回答
            outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
            generated = outputs[0].outputs[0].text.strip().lower()

            # 保存结果
            result = {
                "id": sample.get("id"),
                "task": sample.get("task", ""),
                "question": question,
                "answer": generated,
                "Ground Truth": sample["conversations"][1]["value"].lower(),
                "file_info": sample.get("file_info", {}),
                "image": sample.get("image", []),  # 保存图片路径信息
            }
            results.append(result)

        # 保存推理结果
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(f"推理完成，结果保存到: {output_file}")
        return results

    except Exception as e:
        logger.exception(f"推理过程中出错: {str(e)}")
        return []


def exact_match_by_actions(pred_actions, gt_actions):
    """基于原子动作和参数的精确匹配，忽略空格等因素"""
    if not pred_actions or not gt_actions:
        return False

    # 动作数量必须相同
    if len(pred_actions) != len(gt_actions):
        return False

    # 逐个比较每个动作的名称和参数
    for pred_action, gt_action in zip(pred_actions, gt_actions):
        # 比较动作名称（忽略大小写和空格）
        pred_name = pred_action[0].strip().lower()
        gt_name = gt_action[0].strip().lower()
        if pred_name != gt_name:
            return False

        # 比较参数（忽略参数内部和周围的所有空格，但保留参数顺序）
        # 规范化参数：移除所有空格并转为小写
        pred_params = [re.sub(r"\s+", "", p.strip()).lower() for p in pred_action[1]]
        gt_params = [re.sub(r"\s+", "", p.strip()).lower() for p in gt_action[1]]

        # 参数数量必须相同
        if len(pred_params) != len(gt_params):
            return False

        # 逐个比较参数（已规范化，移除所有空格）
        for pred_param, gt_param in zip(pred_params, gt_params):
            if pred_param != gt_param:
                return False

    return True


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


def step_by_step_match(pred_actions, gt_actions):
    """逐步匹配：从前向后比较，直到第一个不匹配的动作，计算正确比例"""
    if not pred_actions or not gt_actions:
        return 0.0

    min_len = min(len(pred_actions), len(gt_actions))
    correct_count = 0

    for i in range(min_len):
        pred_action = pred_actions[i]
        gt_action = gt_actions[i]

        # 比较动作名称和参数
        if pred_action[0] == gt_action[0] and pred_action[1] == gt_action[1]:
            correct_count += 1
        else:
            break

    return correct_count / len(gt_actions) if gt_actions else 0.0


def parse_plan(plan_text: str):
    """解析 plan 为 (action, params[]) 列表"""
    pattern = r"([A-Za-z]+)\(([^)]*)\)"
    actions = []
    for name, params in re.findall(pattern, plan_text):
        params = [p.strip() for p in params.split(",") if p.strip()]
        actions.append((name.strip(), params))
    return actions


def parameter_score(pred_params, gt_params):
    """参数相似度，完全一致1，不同则按偏差百分比计算"""
    if not gt_params:
        return 1.0 if not pred_params else 0.0

    scores = []
    for i, gt in enumerate(gt_params):
        if i < len(pred_params):
            scores.append(single_param_score(pred_params[i], gt))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def single_param_score(pred, gt):
    """改进版：文字部分匹配 + 数字部分按偏差计算"""
    pred, gt = pred.strip().lower(), gt.strip().lower()

    # 完全一致
    if pred == gt:
        return 1.0

    # 提取数字与文本
    num_pred = extract_number(pred)
    num_gt = extract_number(gt)

    text_pred = re.sub(r"[\d\.°％%]+", "", pred).strip()
    text_gt = re.sub(r"[\d\.°％%]+", "", gt).strip()

    text_match = (
        1.0
        if (text_pred and text_gt and (text_pred in text_gt or text_gt in text_pred))
        else 0.0
    )

    num_score = 1.0
    if num_pred is not None and num_gt is not None and num_gt != 0:
        diff = abs(num_pred - num_gt)
        percent = min(diff / abs(num_gt), 1.0)
        num_score = round(1 - percent, 4)

    if text_pred or text_gt:
        return round(0.5 * text_match + 0.5 * num_score, 4)
    else:
        return num_score


def extract_number(s: str):
    """从字符串中提取数值（如80°C -> 80）"""
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None


def evaluate_pl_results(
    inference_results, test_folder, model_name, custom_result_dir=None
):
    """评估推理结果"""
    appliance_exact_scores = defaultdict(list)
    appliance_step_scores = defaultdict(list)
    file_results = []

    for result in inference_results:
        file_info = result.get("file_info", {})
        appliance_type = file_info.get("appliance_type", "unknown")

        predicted_plan = result.get("answer", "").strip()
        ground_truth = result.get("Ground Truth", "").strip()

        scores = score_plan(predicted_plan, ground_truth)

        file_result = {
            "file": file_info.get("filename", "unknown"),
            "code": file_info.get("code", "unknown"),
            "appliance_type": appliance_type,
            "language": file_info.get("language", "unknown"),
            "exact_match": scores["exact_match"],
            "step_match": scores["step_match"],
            "predicted": predicted_plan,
            "ground_truth": ground_truth,
        }

        file_results.append(file_result)
        appliance_exact_scores[appliance_type].append(scores["exact_match"])
        appliance_step_scores[appliance_type].append(scores["step_match"])

    generate_report(
        file_results,
        appliance_exact_scores,
        appliance_step_scores,
        test_folder,
        model_name,
        custom_result_dir,
    )


def generate_report(
    file_results,
    appliance_exact_scores,
    appliance_step_scores,
    test_folder,
    model_name,
    custom_result_dir=None,
):
    """生成评估报告"""
    report = []
    report.append("=" * 80)
    report.append("PLAN EVALUATION REPORT - DUAL EVALUATION")
    report.append("=" * 80)
    report.append(f"Model: {model_name}")
    report.append(f"Test Folder: {test_folder}")
    report.append(f"Total Files: {len(file_results)}")
    report.append(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 按家电类型统计
    report.append("BY APPLIANCE TYPE:")
    report.append("-" * 60)
    report.append(
        f"{'Appliance Type':<15} {'Exact Match':<12} {'Step Match':<12} {'Count':<8}"
    )
    report.append("-" * 60)

    for appliance_type in sorted(appliance_exact_scores.keys()):
        exact_scores = appliance_exact_scores[appliance_type]
        step_scores = appliance_step_scores[appliance_type]

        exact_avg = sum(exact_scores) / len(exact_scores) if exact_scores else 0.0
        step_avg = sum(step_scores) / len(step_scores) if step_scores else 0.0
        count = len(exact_scores)

        report.append(
            f"{appliance_type:<15} {exact_avg:<12.3f} {step_avg:<12.3f} {count:<8}"
        )

    report.append("")

    # 总体统计
    all_exact_scores = [result["exact_match"] for result in file_results]
    all_step_scores = [result["step_match"] for result in file_results]

    overall_exact_avg = (
        sum(all_exact_scores) / len(all_exact_scores) if all_exact_scores else 0.0
    )
    overall_step_avg = (
        sum(all_step_scores) / len(all_step_scores) if all_step_scores else 0.0
    )

    perfect_exact_matches = sum(1 for score in all_exact_scores if score == 1.0)
    perfect_step_matches = sum(1 for score in all_step_scores if score == 1.0)

    report.append("OVERALL STATISTICS:")
    report.append("-" * 60)
    report.append(f"Overall Exact Match Average: {overall_exact_avg:.3f}")
    report.append(f"Overall Step Match Average:  {overall_step_avg:.3f}")
    report.append(
        f"Perfect Exact Matches (1.0): {perfect_exact_matches}/{len(all_exact_scores)} "
        f"({(perfect_exact_matches/len(all_exact_scores)*100 if all_exact_scores else 0):.1f}%)"
    )
    report.append(
        f"Perfect Step Matches (1.0):  {perfect_step_matches}/{len(all_step_scores)} "
        f"({(perfect_step_matches/len(all_step_scores)*100 if all_step_scores else 0):.1f}%)"
    )
    report.append("")

    # 详细结果
    report.append("DETAILED RESULTS:")
    report.append("-" * 80)
    report.append(f"{'File':<25} {'Appliance':<12} {'Exact':<8} {'Step':<8}")
    report.append("-" * 80)

    for result in sorted(
        file_results, key=lambda x: (x["appliance_type"], -x["exact_match"])
    ):
        report.append(
            f"{result['file']:<25} {result['appliance_type']:<12} "
            f"{result['exact_match']:<8.3f} {result['step_match']:<8.3f}"
        )

    # 保存报告到文件
    if custom_result_dir:
        save_base = custom_result_dir
    else:
        save_base = f"{root}manual_data/Train/{model_name}/result/"
    os.makedirs(save_base, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(save_base) / f"dual_evaluation_report_{timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))
    logger.info(f"双测评报告保存到: {report_file}")

    # 生成汇总统计文件
    generate_summary_report(
        file_results,
        appliance_exact_scores,
        appliance_step_scores,
        model_name,
        timestamp,
        custom_result_dir,
    )


def generate_summary_report(
    file_results,
    appliance_exact_scores,
    appliance_step_scores,
    model_name,
    timestamp,
    custom_result_dir=None,
):
    """生成汇总统计报告"""
    summary = []
    summary.append("=" * 100)
    summary.append("SUMMARY STATISTICS BY APPLIANCE TYPE")
    summary.append("=" * 100)
    summary.append(f"Model: {model_name}")
    summary.append(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")

    # 按家电类型汇总
    summary.append("APPLIANCE TYPE SUMMARY:")
    summary.append("-" * 80)
    summary.append(
        f"{'Appliance Type':<20} {'Exact Match Avg':<15} {'Step Match Avg':<15} "
        f"{'Count':<8} {'Exact Perfect':<12} {'Step Perfect':<12}"
    )
    summary.append("-" * 80)

    for appliance_type in sorted(appliance_exact_scores.keys()):
        exact_scores = appliance_exact_scores[appliance_type]
        step_scores = appliance_step_scores[appliance_type]

        exact_avg = sum(exact_scores) / len(exact_scores) if exact_scores else 0.0
        step_avg = sum(step_scores) / len(step_scores) if step_scores else 0.0
        count = len(exact_scores)

        exact_perfect = sum(1 for score in exact_scores if score == 1.0)
        step_perfect = sum(1 for score in step_scores if score == 1.0)

        summary.append(
            f"{appliance_type:<20} {exact_avg:<15.3f} {step_avg:<15.3f} "
            f"{count:<8} {exact_perfect:<12} {step_perfect:<12}"
        )

    summary.append("")

    # 总体汇总
    all_exact_scores = [result["exact_match"] for result in file_results]
    all_step_scores = [result["step_match"] for result in file_results]

    overall_exact_avg = (
        sum(all_exact_scores) / len(all_exact_scores) if all_exact_scores else 0.0
    )
    overall_step_avg = (
        sum(all_step_scores) / len(all_step_scores) if all_step_scores else 0.0
    )
    total_count = len(file_results)

    overall_exact_perfect = sum(1 for score in all_exact_scores if score == 1.0)
    overall_step_perfect = sum(1 for score in all_step_scores if score == 1.0)

    summary.append("OVERALL SUMMARY:")
    summary.append("-" * 80)
    summary.append(f"Total Files: {total_count}")
    summary.append(f"Overall Exact Match Average: {overall_exact_avg:.3f}")
    summary.append(f"Overall Step Match Average:  {overall_step_avg:.3f}")
    summary.append(
        f"Overall Exact Perfect: {overall_exact_perfect}/{total_count} "
        f"({(overall_exact_perfect/total_count*100 if total_count else 0):.1f}%)"
    )
    summary.append(
        f"Overall Step Perfect:  {overall_step_perfect}/{total_count} "
        f"({(overall_step_perfect/total_count*100 if total_count else 0):.1f}%)"
    )

    if custom_result_dir:
        save_base = custom_result_dir
    else:
        save_base = f"{root}manual_data/Train/{model_name}/result/"
    os.makedirs(save_base, exist_ok=True)

    summary_file = Path(save_base) / f"summary_report_{timestamp}.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    logger.info(f"汇总统计报告保存到: {summary_file}")


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
                print(f"[WARN] 第 {line_no} 行 JSON 解析失败：{e}")
                continue

            total += 1
            pred = data.get("answer", "").strip()
            gt = data.get("Ground Truth", "").strip()

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

    print(f"Total: {total}")
    print(f"Exact Match Accuracy: {exact_acc:.4f}")
    print(f"Step Match Accuracy:  {step_acc:.4f}")

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

    print(f"All results saved to: {txt_path.absolute()}")


# ========================= 关键：单实例进程处理“多文件” =========================


def process_file_list_single_instance(
    model_name, json_file_list, gpu_id, custom_result_dir=None
):
    """
    在独立进程中处理一批JSON文件，固定到指定GPU；
    该进程仅加载一个模型实例，并顺序处理分配到它的文件列表。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # 路径配置
        if os.path.isabs(model_name):
            model_path = model_name
        else:
            model_path = f"{root}ft/scripts/Model_ftd/{model_name}"

        save_base = (
            custom_result_dir
            if custom_result_dir
            else f"{root}manual_data/Train/{model_name}/result/"
        )
        os.makedirs(save_base, exist_ok=True)

        logger.info(
            f"[GPU {gpu_id}] Loading model ONCE for {len(json_file_list)} files: {model_name}"
        )
        processor = AutoProcessor.from_pretrained(model_path)
        # 单卡单模型模式，可以使用更多内存
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

                for sample in test_data:
                    sample["file_info"] = {
                        "filename": json_file.name,
                        "code": code,
                        "appliance_type": appliance_type,
                        "language": language,
                    }

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = (
                    Path(save_base) / f"{json_file.stem}_inference_{timestamp}.jsonl"
                )

                logger.info(f"[GPU {gpu_id}] Processing file: {json_file.name}")
                results = run_pl_inference(llm, processor, test_data, output_file)
                all_results.extend(results)
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


# ========================= 主函数：任务切分与并发 =========================


def main():
    """主函数 - 单机八卡、每卡单模型"""
    parser = argparse.ArgumentParser(
        description="Plan testing system - Single Node 8 GPU, 1 instance per GPU"
    )
    parser.add_argument("--model_name", type=str, help="Model name to test")
    parser.add_argument(
        "--data_path", type=str, help="Directory that contains *.json test files"
    )
    parser.add_argument("--root_path", type=str, help="Root path")
    parser.add_argument(
        "--result_dir", type=str, help="Result directory for saving evaluation results"
    )
    args = parser.parse_args()

    # 命令行参数覆盖
    if args.model_name:
        global model_set
        model_set = [args.model_name]
        logger.info(f"Using model from command line: {args.model_name}")

    if args.data_path:
        global data_base
        data_base = args.data_path

    if args.root_path:
        global root
        root = args.root_path

    custom_result_dir = args.result_dir if args.result_dir else None

    logger.info("Starting plan testing system - Single Node 8 GPU Mode")
    logger.info("Single-instance mode: 1 model instance per GPU")
    logger.info(f"Available GPUs: {gpu_list} (count={len(gpu_list)})")
    logger.info(f"Model to process: {model_set[0] if model_set else 'None'}")
    if data_base:
        logger.info(f"Data path: {data_base}")

    start_time = time.time()

    # 只处理第一个模型（每次只测评一个模型）
    if len(model_set) == 0:
        logger.error("No model specified")
        return
    model_name = model_set[0]

    # 确定数据路径
    if data_base:
        data_base_path = data_base
    else:
        data_base_path = f"{root}manual_data/Train/{model_name}/test_data/"

    if not os.path.exists(data_base_path):
        logger.error(f"Test data path not found: {data_base_path}")
        return

    # 收集所有JSON文件
    json_files = list(Path(data_base_path).glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {data_base_path}")
        return

    logger.info(f"Found {len(json_files)} JSON files to process")

    # ---- 单卡单模型：文件轮询分配给8张GPU ----
    max_workers = len(gpu_list)  # 8个进程，每张GPU一个

    # 把文件轮询分配给8张GPU
    gpu_file_lists = [[] for _ in range(len(gpu_list))]
    for idx, jf in enumerate(json_files):
        gpu_idx = idx % len(gpu_list)
        gpu_file_lists[gpu_idx].append(jf)

    # 打印GPU文件分配
    for gpu_idx, file_list in enumerate(gpu_file_lists):
        if file_list:
            logger.info(f"GPU {gpu_list[gpu_idx]}: {len(file_list)} files")

    # ---- 并发提交：每张GPU一个进程 ----
    from concurrent.futures import ProcessPoolExecutor, as_completed

    logger.info(
        f"Starting parallel execution with {max_workers} workers (1 instance per GPU)."
    )

    all_results_by_model = defaultdict(dict)
    completed_count = 0
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for gpu_idx in range(len(gpu_list)):
            if not gpu_file_lists[gpu_idx]:
                continue
            gpu_id = gpu_list[gpu_idx]
            futures.append(
                executor.submit(
                    process_file_list_single_instance,
                    model_name,
                    gpu_file_lists[gpu_idx],
                    gpu_id,
                    custom_result_dir,
                )
            )

        total_tasks = len(futures)
        for fut in as_completed(futures):
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
                logger.info(f"✅ [{completed_count}/{total_tasks}] GPU finished.")
            except Exception as e:
                logger.error(f"❌ [{completed_count}/{total_tasks}] GPU failed: {e}")

    # 生成最终评估报告
    logger.info("\n" + "=" * 80)
    logger.info("Generating final evaluation report...")

    if model_name in all_results_by_model:
        results = all_results_by_model[model_name].get("results", [])
        data_dir = all_results_by_model[model_name].get("data_dir", data_base_path)
        if results:
            logger.info(
                f"Generating report for model: {model_name} ({len(results)} samples)"
            )
            evaluate_pl_results(results, str(data_dir), model_name, custom_result_dir)
        else:
            logger.warning("No results collected for evaluation")
    else:
        logger.warning("No results collected for model")

    duration = time.time() - start_time
    total_tasks_done = completed_count
    logger.info(f"\n{'='*80}")
    logger.info(
        f"All tasks completed in {duration:.2f} seconds ({duration/60:.2f} minutes)"
    )
    if total_tasks_done:
        logger.info(f"Average time per slot: {duration/total_tasks_done:.2f} seconds")
    logger.info(f"Total files processed: {len(json_files)}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
