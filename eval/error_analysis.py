#!/usr/bin/env python3
"""
错误分析脚本：分析测评结果中的错误条数和错误模式
通过调用脚本方式，传递文件夹绝对路径参数
"""
import os
import sys
import json
from pathlib import Path
import logging
from datetime import datetime

# 导入错误检测函数
from find_error import find_error_steps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_errors(result_dir, config_dir, iteration):
    """
    分析测评结果中的错误

    Args:
        result_dir: 测评结果目录的绝对路径
        config_dir: 配置目录的绝对路径
        iteration: 当前迭代次数

    Returns:
        dict: 错误分析结果
    """
    logger.info(f"开始错误分析:")
    logger.info(f"  结果目录: {result_dir}")
    logger.info(f"  配置目录: {config_dir}")
    logger.info(f"  迭代次数: {iteration}")

    total_errors = 0
    total_samples = 0
    error_samples = []

    # 查找推理结果文件
    result_files = list(Path(result_dir).glob("*_inference_*.jsonl"))

    if not result_files:
        logger.warning(f"未找到推理结果文件: {result_dir}")
        return {
            "total_samples": 0,
            "total_errors": 0,
            "error_rate": 0.0,
            "error_samples": [],
            "iteration": iteration,
        }

    logger.info(f"找到 {len(result_files)} 个推理结果文件")

    for result_file in result_files:
        logger.info(f"处理文件: {result_file.name}")

        with open(result_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        total_samples += 1
                        predicted = data.get("answer", "").strip()
                        ground_truth = data.get("Ground Truth", "").strip()

                        # 使用find_error.py分析错误步骤
                        error_steps = find_error_steps(ground_truth, predicted)

                        if error_steps:
                            total_errors += len(error_steps)
                            error_sample = {
                                "id": data.get("id", ""),
                                "predicted": predicted,
                                "ground_truth": ground_truth,
                                "error_steps": error_steps,
                                "error_count": len(error_steps),
                                "file_info": data.get("file_info", {}),
                                "line_number": line_no,
                            }
                            error_samples.append(error_sample)

                            logger.debug(
                                f"样本 {data.get('id', '')} 有 {len(error_steps)} 个错误步骤: {error_steps}"
                            )

                    except json.JSONDecodeError as e:
                        logger.warning(f"解析JSON失败 (行 {line_no}): {e}")
                    except Exception as e:
                        logger.error(f"处理样本失败 (行 {line_no}): {e}")

    # 计算错误率
    error_rate = total_errors / total_samples if total_samples > 0 else 0.0

    # 生成错误分析结果
    error_analysis = {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "error_rate": error_rate,
        "error_samples": error_samples,
        "iteration": iteration,
        "result_dir": result_dir,
        "config_dir": config_dir,
    }

    # 保存错误分析结果
    error_analysis_file = os.path.join(
        config_dir, f"error_analysis_iter_{iteration}.json"
    )
    with open(error_analysis_file, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)

    logger.info(f"错误分析完成:")
    logger.info(f"  总样本数: {total_samples}")
    logger.info(f"  总错误数: {total_errors}")
    logger.info(f"  错误率: {error_rate:.3f}")
    logger.info(f"  错误样本数: {len(error_samples)}")
    logger.info(f"  结果保存到: {error_analysis_file}")

    return error_analysis


def generate_error_summary(error_analysis):
    """
    生成错误摘要报告

    Args:
        error_analysis: 错误分析结果

    Returns:
        str: 错误摘要报告
    """
    total_samples = error_analysis["total_samples"]
    total_errors = error_analysis["total_errors"]
    error_rate = error_analysis["error_rate"]
    error_samples = error_analysis["error_samples"]
    iteration = error_analysis["iteration"]

    summary = []
    summary.append("=" * 100)
    summary.append(f"错误分析摘要报告 - 迭代 {iteration}")
    summary.append("=" * 100)
    summary.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"总样本数: {total_samples}")
    summary.append(f"总错误数: {total_errors}")
    summary.append(f"错误率: {error_rate:.3f} ({error_rate*100:.1f}%)")
    summary.append(f"错误样本数: {len(error_samples)}")
    summary.append(f"正确样本数: {total_samples - len(error_samples)}")
    summary.append("")

    if error_samples:
        summary.append("错误样本详情:")
        summary.append("-" * 100)

        # 按错误数量排序
        error_samples_sorted = sorted(
            error_samples, key=lambda x: x["error_count"], reverse=True
        )

        for i, sample in enumerate(error_samples_sorted[:20]):  # 显示前20个
            summary.append(f"{i+1}. 样本ID: {sample['id']}")
            summary.append(f"   错误数量: {sample['error_count']}")
            summary.append(f"   错误步骤: {sample['error_steps']}")
            summary.append(
                f"   文件信息: {sample.get('file_info', {}).get('filename', 'N/A')}"
            )
            summary.append(f"   预测结果:")
            summary.append(f"     {sample['predicted']}")
            summary.append(f"   标准答案:")
            summary.append(f"     {sample['ground_truth']}")
            summary.append("")

        if len(error_samples) > 20:
            summary.append(f"... 还有 {len(error_samples) - 20} 个错误样本")

    summary.append("=" * 100)

    return "\n".join(summary)


def generate_detailed_error_log(error_analysis):
    """
    生成详细的错误分析日志

    Args:
        error_analysis: 错误分析结果

    Returns:
        str: 详细错误日志
    """
    total_samples = error_analysis["total_samples"]
    total_errors = error_analysis["total_errors"]
    error_rate = error_analysis["error_rate"]
    error_samples = error_analysis["error_samples"]
    iteration = error_analysis["iteration"]

    log = []
    log.append("=" * 120)
    log.append(f"详细错误分析日志 - 迭代 {iteration}")
    log.append("=" * 120)
    log.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append(f"结果目录: {error_analysis.get('result_dir', 'N/A')}")
    log.append(f"配置目录: {error_analysis.get('config_dir', 'N/A')}")
    log.append("")

    # 统计信息
    log.append("统计信息:")
    log.append("-" * 120)
    log.append(f"总样本数: {total_samples}")
    log.append(f"总错误数: {total_errors}")
    log.append(f"错误率: {error_rate:.3f} ({error_rate*100:.1f}%)")
    log.append(f"错误样本数: {len(error_samples)}")
    log.append(f"正确样本数: {total_samples - len(error_samples)}")
    log.append("")

    if error_samples:
        # 错误步骤统计
        error_step_counts = {}
        for sample in error_samples:
            for step in sample["error_steps"]:
                error_step_counts[step] = error_step_counts.get(step, 0) + 1

        log.append("错误步骤统计:")
        log.append("-" * 120)
        for step, count in sorted(error_step_counts.items()):
            log.append(f"步骤 {step}: {count} 次错误")
        log.append("")

        # 错误样本详情
        log.append("所有错误样本详情:")
        log.append("-" * 120)

        # 按错误数量排序
        error_samples_sorted = sorted(
            error_samples, key=lambda x: x["error_count"], reverse=True
        )

        for i, sample in enumerate(error_samples_sorted):
            log.append(f"样本 {i+1}/{len(error_samples)}:")
            log.append(f"  样本ID: {sample['id']}")
            log.append(f"  错误数量: {sample['error_count']}")
            log.append(f"  错误步骤: {sample['error_steps']}")
            log.append(f"  文件信息: {sample.get('file_info', {})}")
            log.append(f"  行号: {sample.get('line_number', 'N/A')}")
            log.append(f"  预测结果:")
            log.append(f"    {sample['predicted']}")
            log.append(f"  标准答案:")
            log.append(f"    {sample['ground_truth']}")
            log.append("")

    log.append("=" * 120)
    log.append(f"日志生成完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("=" * 120)

    return "\n".join(log)


def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("用法: python error_analysis.py <result_dir> <config_dir> <iteration>")
        print("参数说明:")
        print("  result_dir: 测评结果目录的绝对路径")
        print("  config_dir: 配置目录的绝对路径")
        print("  iteration: 当前迭代次数")
        sys.exit(1)

    result_dir = sys.argv[1]
    config_dir = sys.argv[2]
    iteration = int(sys.argv[3])

    # 检查参数
    if not os.path.exists(result_dir):
        print(f"错误: 结果目录不存在: {result_dir}")
        sys.exit(1)

    if not os.path.exists(config_dir):
        print(f"错误: 配置目录不存在: {config_dir}")
        sys.exit(1)

    try:
        # 执行错误分析
        error_analysis = analyze_errors(result_dir, config_dir, iteration)

        # 生成错误摘要报告
        summary = generate_error_summary(error_analysis)

        # 确保 config_dir 存在（config_dir 就是 log 目录）
        os.makedirs(config_dir, exist_ok=True)

        # 保存摘要报告到 config_dir（就是 log 目录）
        summary_file = os.path.join(config_dir, f"error_summary_iter_{iteration}.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)

        # 生成详细错误日志
        detailed_log = generate_detailed_error_log(error_analysis)

        # 保存详细错误日志
        detailed_log_file = os.path.join(
            config_dir, f"error_detailed_log_iter_{iteration}.txt"
        )
        with open(detailed_log_file, "w", encoding="utf-8") as f:
            f.write(detailed_log)

        # 打印摘要报告
        print(summary)
        print(f"\n错误摘要报告保存到: {summary_file}")
        print(f"详细错误日志保存到: {detailed_log_file}")

        # 返回错误分析结果
        print(f"\n错误分析结果:")
        print(f"  总样本数: {error_analysis['total_samples']}")
        print(f"  总错误数: {error_analysis['total_errors']}")
        print(f"  错误率: {error_analysis['error_rate']:.3f}")
        print(f"  错误样本数: {len(error_analysis['error_samples'])}")

    except Exception as e:
        logger.error(f"错误分析失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
