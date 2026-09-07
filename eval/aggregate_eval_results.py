#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多机推理结果汇总统计脚本
用于合并所有节点的推理结果并生成汇总统计报告

用法:
python aggregate_eval_results.py \
    --test_script test.py \
    --result_dir /path/to/result \
    --test_data_path /path/to/test/data \
    --model_name /path/to/model \
    --eval_name "plan" \
    --root_path 
"""

import os
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def aggregate_plan_results(result_dir, test_data_path, model_name, root_path):
    """汇总plan测试结果"""
    logger.info("开始汇总plan测试结果...")

    # 添加路径到sys.path
    sys.path.insert(0, f"{root_path}manual_data/Data_Flywheel")
    from test import evaluate_pl_results

    # 收集所有jsonl结果文件
    result_dir = Path(result_dir)
    jsonl_files = list(result_dir.glob("*_inference_*.jsonl"))

    if not jsonl_files:
        logger.warning("未找到结果文件")
        return False

    logger.info(f"找到 {len(jsonl_files)} 个结果文件，开始合并统计...")

    # 读取所有结果
    all_results = []
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
        except Exception as e:
            logger.error(f"读取文件 {jsonl_file} 失败: {e}")

    logger.info(f"总共收集到 {len(all_results)} 条结果")

    if not all_results:
        logger.warning("没有收集到任何结果")
        return False

    # 调用评估函数生成报告
    evaluate_pl_results(all_results, test_data_path, model_name, str(result_dir))
    logger.info("✓ plan测试汇总统计报告已生成")
    return True


def aggregate_bbox_results(result_dir):
    """汇总bbox测试结果"""
    logger.info("开始汇总bbox测试结果...")

    result_dir = Path(result_dir)
    jsonl_files = list(result_dir.glob("*_inference_*.jsonl"))

    if not jsonl_files:
        logger.warning("未找到结果文件")
        return False

    # 读取所有结果并统计，同时生成合并文件
    total = 0
    valid = 0
    iou_sum = 0.0
    dist_sum = 0.0
    above_05 = 0
    merged_jsonl = result_dir / "merged_bbox_results.jsonl"

    try:
        with open(merged_jsonl, "w", encoding="utf-8") as merged_f:
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            merged_f.write(stripped + "\n")
                            try:
                                data = json.loads(stripped)
                            except json.JSONDecodeError as e:
                                logger.warning(f"解析 JSON 失败 ({jsonl_file}): {e}")
                                continue
                            total += 1
                            if data.get("IoU") is not None and data.get("gt_bbox"):
                                valid += 1
                                iou = data["IoU"]
                                dist = data.get("center_distance", 0)
                                iou_sum += iou
                                dist_sum += dist
                                if iou >= 0.5:
                                    above_05 += 1
                except Exception as e:
                    logger.error(f"读取文件 {jsonl_file} 失败: {e}")
    except Exception as e:
        logger.error(f"写入合并文件 {merged_jsonl} 失败: {e}")
        return False

    logger.info(f"已合并所有结果文件到: {merged_jsonl}")

    logger.info(f"总共处理 {total} 个样本，有效bbox对: {valid}")

    # 生成汇总报告
    summary_file = result_dir / "bbox_summary_report.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BBox Evaluation Summary Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Valid bbox pairs: {valid}\n")
        if valid > 0:
            f.write(f"Average IoU: {iou_sum/valid:.4f}\n")
            f.write(f"Average center distance: {dist_sum/valid:.2f} px\n")
            f.write(f"IoU >= 0.5 ratio: {above_05}/{valid} ({above_05/valid:.2%})\n")
        f.write("=" * 80 + "\n")

    logger.info(f"✓ bbox测试汇总统计报告已保存到: {summary_file}")
    return True


def aggregate_keypage_results(result_dir, test_data_path, model_name, root_path):
    """汇总keypage测试结果"""
    logger.info("开始汇总keypage测试结果...")

    # 添加路径到sys.path
    sys.path.insert(0, f"{root_path}manual_data/Data_Flywheel")
    from test_keypage import evaluate_keypage_results

    # 收集所有jsonl结果文件
    result_dir = Path(result_dir)
    jsonl_files = list(result_dir.glob("*_inference_*.jsonl"))

    if not jsonl_files:
        logger.warning("未找到结果文件")
        return False

    logger.info(f"找到 {len(jsonl_files)} 个结果文件，开始合并统计...")

    # 读取所有结果
    all_results = []
    merged_jsonl = result_dir / "merged_keypage_results.jsonl"
    try:
        with open(merged_jsonl, "w", encoding="utf-8") as merged_f:
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            merged_f.write(stripped + "\n")
                            all_results.append(json.loads(stripped))
                except Exception as e:
                    logger.error(f"读取文件 {jsonl_file} 失败: {e}")
    except Exception as e:
        logger.error(f"写入 merged_keypage_results.jsonl 失败: {e}")

    logger.info(f"总共收集到 {len(all_results)} 条结果")
    logger.info(f"已合并所有结果文件到: {merged_jsonl}")

    if not all_results:
        logger.warning("没有收集到任何结果")
        return False

    # 调用评估函数生成报告
    evaluate_keypage_results(all_results, test_data_path, model_name, str(result_dir))
    logger.info("✓ keypage测试汇总统计报告已生成")
    return True


def aggregate_close_loop_results(result_dir, root_path):
    """汇总close_loop测试结果"""
    logger.info("开始汇总close_loop测试结果...")

    # 添加路径到sys.path
    sys.path.insert(0, f"{root_path}manual_data/Data_Flywheel")
    from test_close_loop import test_totally_same

    # 收集所有jsonl结果文件
    result_dir = Path(result_dir)
    jsonl_files = list(result_dir.glob("*_inference_*.jsonl"))

    if not jsonl_files:
        logger.warning("未找到结果文件")
        return False

    logger.info(f"找到 {len(jsonl_files)} 个结果文件，开始合并统计...")

    # 合并所有jsonl文件到一个临时文件
    merged_jsonl = result_dir / "merged_close_loop_results.jsonl"
    with open(merged_jsonl, "w", encoding="utf-8") as merged_f:
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            merged_f.write(line)
            except Exception as e:
                logger.error(f"读取文件 {jsonl_file} 失败: {e}")

    logger.info(f"已合并所有结果文件到: {merged_jsonl}")

    # 调用test_totally_same计算统计
    test_totally_same(merged_jsonl)
    logger.info("✓ close_loop测试汇总统计报告已生成")
    return True


def main():
    parser = argparse.ArgumentParser(description="多机推理结果汇总统计脚本")
    parser.add_argument(
        "--test_script",
        type=str,
        required=True,
        help="测试脚本名称 (test.py, test_bbox.py, test_keypage.py, test_close_loop.py)",
    )
    parser.add_argument("--result_dir", type=str, required=True, help="结果目录路径")
    parser.add_argument("--test_data_path", type=str, default="", help="测试数据路径")
    parser.add_argument("--model_name", type=str, default="", help="模型名称或路径")
    parser.add_argument(
        "--root_path",
        type=str,
        default="",
        help="根路径",
    )
    parser.add_argument(
        "--eval_name", type=str, default="", help="评测名称（用于日志）"
    )

    args = parser.parse_args()

    logger.info(f"开始汇总{args.eval_name or args.test_script}测试结果...")
    logger.info(f"结果目录: {args.result_dir}")

    success = False

    # 根据不同的测试脚本，调用相应的汇总函数
    if args.test_script == "test.py":
        success = aggregate_plan_results(
            args.result_dir, args.test_data_path, args.model_name, args.root_path
        )
    elif args.test_script == "test_bbox.py":
        success = aggregate_bbox_results(args.result_dir)
    elif args.test_script == "test_keypage.py":
        success = aggregate_keypage_results(
            args.result_dir, args.test_data_path, args.model_name, args.root_path
        )
    elif args.test_script == "test_close_loop.py":
        success = aggregate_close_loop_results(args.result_dir, args.root_path)
    else:
        logger.error(f"不支持的测试脚本: {args.test_script}")
        return 1

    if success:
        logger.info(f"✓ {args.eval_name or args.test_script}汇总统计完成")
        return 0
    else:
        logger.error(f"✗ {args.eval_name or args.test_script}汇总统计失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
