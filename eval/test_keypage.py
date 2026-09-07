#!/usr/bin/env python3
"""
Keypage测试脚本 - 单卡单模型推理
用于测试页面是否全面介绍家电组件的二分类任务
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import numpy as np

# Optional dependencies:
# - sklearn: used for classification metrics. If missing, we fall back to pure-Python metrics.

HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    HAS_MATPLOTLIB = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("keypage_test.log"),
    ],
)
logger = logging.getLogger(__name__)

# 全局变量
model_set = []
data_base = ""
root = ""

# GPU配置
gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]  # 8张GPU


def load_model(model_name: str, gpu_id: int):
    """加载模型"""
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        logger.info(f"Loading model {model_name} on GPU {gpu_id}")

        # 设置GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 加载模型
        model = LLM(
            model=model_name,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            trust_remote_code=True,
        )

        # 加载processor（处理多模态输入）
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        logger.info(f"✓ Model {model_name} loaded successfully on GPU {gpu_id}")
        return model, processor

    except Exception as e:
        logger.error(f"Failed to load model {model_name} on GPU {gpu_id}: {e}")
        return None, None


def process_images(img_paths):
    """处理图像并返回处理后的图像列表和尺寸信息（保持原图尺寸）"""
    from PIL import Image, ImageOps

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
            # 创建占位图像
            placeholder = Image.new("RGB", (224, 224), (255, 255, 255))
            images.append(placeholder)
            image_sizes.append(placeholder.size)
    return images, image_sizes


def run_keypage_inference(
    model, processor, data_item: Dict[str, Any], gpu_id: int
) -> Dict[str, Any]:
    """运行keypage推理"""
    try:
        from vllm import SamplingParams

        # 从数据的 conversations 中提取 from: "human" 的问题
        conversations = data_item.get("conversations", [])
        question = None
        for conv in conversations:
            if conv.get("from") == "human":
                question = conv.get("value", "")
                break

        # 如果找不到 human 的问题，使用默认问题
        if not question:
            logger.warning(
                f"No human question found in data item {data_item.get('id', 'unknown')}, using default question"
            )
            question = "Please judge whether this page comprehensively introduces the appliance components. Answer with 'yes' or 'no' only."

        
        ######################
        #question = "<image>\nAnalyze this manual page image and judge if it is a 'Product Overview' page. Answer 'yes' or 'no' only."


        # 处理图像
        img_paths = data_item.get("image", [])
        if not img_paths:
            logger.warning(f"No images found in data item")
            return None

        # 确保img_paths是列表
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        elif not isinstance(img_paths, list):
            logger.error(f"Invalid image format: {type(img_paths)}")
            return None

        images, _ = process_images(img_paths)

        # 构建消息
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

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": multi_modal_data,
        }

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=10,
            stop=["\n", ".", "!", "?"],
        )

        # 运行推理
        outputs = model.generate([llm_inputs], sampling_params=sampling_params)

        if outputs and len(outputs) > 0:
            # 保存原始文本（转换为yes/no之前的原数据）
            generated_text_raw = outputs[0].outputs[0].text
            # 处理后的文本用于判断
            generated_text = generated_text_raw.strip().lower()

            # 解析预测结果：若包含 product overview 或明确 yes 文字，则归为 yes
            pred_label = (
                "yes"
                if ("product overview" in generated_text or "yes" in generated_text)
                else "no"
            )

            # 获取真实标签
            conversations = data_item.get("conversations", [])
            gt_label = None
            gt_label_raw = None
            for conv in conversations:
                if conv.get("from") == "gpt":
                    gt_label_raw = conv.get("value", "")
                    gt_text = gt_label_raw.strip().lower()
                    gt_label = (
                        "yes"
                        if ("product overview" in gt_text or "yes" in gt_text)
                        else "no"
                    )
                    break

            if gt_label is None:
                logger.warning(f"No ground truth label found")
                return None

            # 构建结果
            result = {
                "id": data_item.get("id", f"item_{int(time.time())}"),
                "image_paths": img_paths,
                "question": question,
                "prediction": pred_label,
                "ground_truth": gt_label,
                "ground_truth_raw": gt_label_raw,
                "generated_text": generated_text,
                "generated_text_raw": generated_text_raw,  # 原始输出文本（转换为yes/no之前）
                "correct": pred_label == gt_label,
                "gpu_id": gpu_id,
            }

            return result
        else:
            logger.warning(f"No output generated for data item")
            return None

    except Exception as e:
        logger.error(f"Error in keypage inference: {e}")
        return None


def process_file_list_single_instance(
    model_name: str, file_list: List[Path], gpu_id: int, result_dir: str
):
    """处理文件列表（单实例模式）"""
    try:
        logger.info(f"Processing {len(file_list)} files on GPU {gpu_id}")

        # 加载模型
        model, processor = load_model(model_name, gpu_id)
        if model is None or processor is None:
            logger.error(f"Failed to load model on GPU {gpu_id}")
            return None

        results = []  # 汇总所有文件的结果（用于最终汇总评估）
        data_dir = None

        for file_path in file_list:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data_dir is None:
                    data_dir = str(file_path.parent)

                # 处理每个数据项（按文件单独收集并输出jsonl）
                per_file_results = []
                for item in data:
                    result = run_keypage_inference(model, processor, item, gpu_id)
                    if result:
                        per_file_results.append(result)
                        results.append(result)  # 也加入到总体结果中

                logger.info(f"Processed {file_path.name} on GPU {gpu_id}")

                # 为当前文件保存独立结果jsonl
                if per_file_results:
                    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                    out_path = os.path.join(
                        result_dir, f"{Path(file_path).stem}_inference_{ts}.jsonl"
                    )
                    os.makedirs(result_dir, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as fout:
                        for r in per_file_results:
                            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    logger.info(
                        f"Saved {len(per_file_results)} results to per-file output: {out_path}"
                    )

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

        # 不再生成按GPU汇总的jsonl；汇总由后续评估步骤产出txt报告

        return model_name, results, data_dir

    except Exception as e:
        logger.error(f"Error in process_file_list_single_instance: {e}")
        return None


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算评估指标"""
    if not results:
        return {}

    # 提取预测和真实标签
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["prediction"] for r in results]

    # 计算基本指标 + 混淆矩阵
    if HAS_SKLEARN:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label="yes", zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label="yes", zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label="yes", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=["no", "yes"])
        cm_list = cm.tolist()
        tn, fp, fn, tp = cm.ravel()
    else:
        # Pure-Python fallback (labels are strictly "no"/"yes")
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == "yes" and p == "yes")
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == "no" and p == "no")
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == "no" and p == "yes")
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == "yes" and p == "no")
        total = len(y_true)
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        cm_list = [[tn, fp], [fn, tp]]

    # 计算每个类别的指标
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "confusion_matrix_array": cm_list,
        "total_samples": len(results),
        "correct_predictions": sum(r["correct"] for r in results),
    }

    return metrics


def plot_confusion_matrix(cm, labels, title, save_path):
    """绘制混淆矩阵（不使用 seaborn）"""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available; skip confusion matrix plotting: %s", save_path)
        return
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_keypage_results(
    results: List[Dict[str, Any]], data_dir: str, model_name: str, result_dir: str
):
    """评估keypage结果并生成报告"""
    if not results:
        logger.warning("No results to evaluate")
        return

    logger.info(f"Evaluating {len(results)} keypage results for model {model_name}")

    # 计算指标
    metrics = calculate_metrics(results)

    if not metrics:
        logger.error("Failed to calculate metrics")
        return

    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)

    # 提取模型名称用于文件名（处理绝对路径的情况）
    if os.path.isabs(model_name):
        model_name_for_file = os.path.basename(model_name.rstrip("/"))
    else:
        model_name_for_file = model_name

    # 按需求：不生成汇总jsonl，仅保存每文件的独立jsonl与最终汇总txt

    # 绘制混淆矩阵
    cm = np.array(metrics["confusion_matrix_array"])
    labels = ["no", "yes"]
    cm_path = os.path.join(
        result_dir, f"keypage_confusion_matrix_{model_name_for_file}.png"
    )
    plot_confusion_matrix(
        cm, labels, f"Keypage Confusion Matrix - {model_name}", cm_path
    )

    # 生成汇总报告
    report_file = os.path.join(
        result_dir, f"keypage_summary_report_{model_name_for_file}.txt"
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Keypage Test Results - {model_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Data Directory: {data_dir}\n")
        f.write(f"Total Samples: {metrics['total_samples']}\n")
        f.write(f"Correct Predictions: {metrics['correct_predictions']}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write("\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  True Negative:  {metrics['confusion_matrix']['true_negative']}\n")
        f.write(f"  False Positive: {metrics['confusion_matrix']['false_positive']}\n")
        f.write(f"  False Negative: {metrics['confusion_matrix']['false_negative']}\n")
        f.write(f"  True Positive:  {metrics['confusion_matrix']['true_positive']}\n")
        f.write("\n")
        f.write("Confusion Matrix Array:\n")
        f.write(f"  {metrics['confusion_matrix_array'][0]}\n")
        f.write(f"  {metrics['confusion_matrix_array'][1]}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")

    logger.info(f"✓ Keypage evaluation completed for {model_name}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Results saved to: {result_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Keypage Test - Single Node 8 GPU Mode"
    )
    parser.add_argument("--model_name", type=str, help="Model name to test")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Directory that contains *.json test files or file list",
    )
    parser.add_argument("--root_path", type=str, help="Root path")
    parser.add_argument(
        "--result_dir", type=str, help="Result directory for saving evaluation results"
    )
    parser.add_argument(
        "--multi_node", action="store_true", help="Enable multi-node inference"
    )
    parser.add_argument(
        "--node_rank", type=int, default=0, help="Node rank for multi-node inference"
    )
    parser.add_argument(
        "--total_nodes", type=int, default=1, help="Total number of nodes"
    )

    # 兼容旧版脚本的参数格式: python test_keypage.py <model> <data_path> <root> [result_dir] [iteration]
    legacy_iteration = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        legacy_args = sys.argv[1:]
        if len(legacy_args) < 2:
            parser.error(
                "Legacy usage: python test_keypage.py <model_name> <data_path> <root_path> [result_dir] [iteration]"
            )

        cli_args = ["--model_name", legacy_args[0], "--data_path", legacy_args[1]]

        if len(legacy_args) >= 3 and legacy_args[2]:
            cli_args.extend(["--root_path", legacy_args[2]])

        if len(legacy_args) >= 4 and legacy_args[3]:
            cli_args.extend(["--result_dir", legacy_args[3]])

        if len(legacy_args) >= 5:
            try:
                legacy_iteration = int(legacy_args[4])
            except ValueError:
                logger.warning(
                    "Invalid legacy iteration value provided: %s", legacy_args[4]
                )

        args = parser.parse_args(cli_args)
        setattr(args, "legacy_iteration", legacy_iteration)
    else:
        args = parser.parse_args()
        setattr(args, "legacy_iteration", None)

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

    logger.info("Starting keypage testing system - Single Node 8 GPU Mode")
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
        data_base_path = f"{root}manual_data/Train/{model_name}/keypage_test_data/"

    if not os.path.exists(data_base_path):
        logger.error(f"Keypage test data path not found: {data_base_path}")
        return

    # 收集JSON文件（支持目录或文件列表）
    json_files = []
    if os.path.isfile(data_base_path) and data_base_path.endswith(".txt"):
        # 文件列表模式（多机推理）
        logger.info(f"Loading file list from: {data_base_path}")
        with open(data_base_path, "r") as f:
            for line in f:
                file_path = line.strip()
                if file_path and os.path.exists(file_path):
                    json_files.append(Path(file_path))
        logger.info(f"Loaded {len(json_files)} files from file list")
    else:
        # 目录模式（单机推理）
        json_files = list(Path(data_base_path).glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in directory")

    if not json_files:
        logger.warning(f"No JSON files found in {data_base_path}")
        return

    logger.info(f"Processing {len(json_files)} JSON files")

    # ---- 单卡单模型：文件轮询分配给8张GPU ----
    max_workers = len(gpu_list)  # 8个进程，每张GPU一个

    # 把文件轮询分配给8张GPU
    gpu_file_lists = [[] for _ in range(len(gpu_list))]
    for idx, jf in enumerate(json_files):
        gpu_idx = idx % len(gpu_list)
        gpu_file_lists[gpu_idx].append(jf)

    # 显示分配情况
    for gpu_idx, file_list in enumerate(gpu_file_lists):
        logger.info(f"GPU {gpu_idx}: {len(file_list)} files")

    # 创建结果目录
    if custom_result_dir:
        result_dir = custom_result_dir
    else:
        result_dir = f"{root}manual_data/Flywheel_Train/keypage_results/{model_name}/"

    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Result directory: {result_dir}")

    # 使用ProcessPoolExecutor并行处理
    all_results_by_model = {}
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for gpu_idx, file_list in enumerate(gpu_file_lists):
            if file_list:  # 只处理有文件的GPU
                futures.append(
                    executor.submit(
                        process_file_list_single_instance,
                        model_name,
                        file_list,
                        gpu_list[gpu_idx],
                        result_dir,
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
    logger.info("Generating final keypage evaluation report...")

    if model_name in all_results_by_model:
        results = all_results_by_model[model_name].get("results", [])
        data_dir = all_results_by_model[model_name].get("data_dir", data_base_path)
        if results:
            logger.info(
                f"Generating report for model: {model_name} ({len(results)} samples)"
            )
            evaluate_keypage_results(results, str(data_dir), model_name, result_dir)
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

        
