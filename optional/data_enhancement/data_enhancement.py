#!/usr/bin/env python3
"""
Plan数据增强脚本
逻辑：
1. 在测评结果目录中查找最新的 summary_report_*.txt
2. 解析 "Overall Exact Match Average" 数值
3. 若低于阈值，则从数据增强目录中每个 JSON 随机抽取一半样本并写入新的训练数据文件
"""

import os
import sys
import json
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXACT_THRESHOLD = float(os.getenv("PLAN_ENHANCEMENT_THRESHOLD", 0.7))
SAMPLE_RATIO = 0.5
RNG_SEED = int(os.getenv("PLAN_ENHANCEMENT_SEED", 42))
SUMMARY_PREFIX = "summary_report_"
OUTPUT_FILENAME_TEMPLATE = "enhanced_plan_iter_{iteration}.json"
SUMMARY_OUTPUT_TEMPLATE = "enhancement_plan_summary_iter_{iteration}.json"


def usage():
    print(
        "用法: python data_enhancement.py <config_dir> <new_data_dir> <result_dir> <iteration>"
    )
    print("参数说明:")
    print("  config_dir: 配置目录的绝对路径（通常为 ITERATION_DIR/log）")
    print("  new_data_dir: 新训练数据目录的绝对路径")
    print("  result_dir: 测评结果目录的绝对路径")
    print("  iteration: 当前迭代次数（整数）")


def resolve_iteration_dir(config_dir: str) -> Path:
    cfg_path = Path(config_dir)
    if cfg_path.name == "log":
        return cfg_path.parent
    return cfg_path


def find_latest_summary(result_dir: Path):
    summary_files = list(result_dir.glob(f"{SUMMARY_PREFIX}*.txt"))
    if not summary_files:
        summary_files = list(result_dir.rglob(f"{SUMMARY_PREFIX}*.txt"))
    if not summary_files:
        return None
    summary_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return summary_files[0]


def parse_exact_match(summary_file: Path):
    try:
        with summary_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Overall Exact Match Average"):
                    try:
                        value = float(line.split(":")[1].strip())
                        return value
                    except (IndexError, ValueError):
                        logger.warning(f"解析 Exact Match 失败: {line}")
                        return None
    except Exception as exc:
        logger.warning(f"读取 {summary_file} 失败: {exc}")
    return None


def collect_random_samples(enhancement_dir: Path):
    rng = random.Random(RNG_SEED)
    collected = []

    if not enhancement_dir.exists():
        logger.warning(f"数据增强目录不存在: {enhancement_dir}")
        return collected

    json_files = [
        p
        for p in enhancement_dir.rglob("*.json")
        if p.is_file() and not p.name.endswith("checkpoint.json")
    ]

    if not json_files:
        logger.warning(f"在 {enhancement_dir} 中未找到 JSON 文件")
        return collected

    logger.info(
        f"准备从 {len(json_files)} 个增强文件中随机提取 {int(SAMPLE_RATIO * 100)}% 样本"
    )

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                content = json.load(f)
                if not isinstance(content, list):
                    content = [content]
                if not content:
                    continue
                sample_size = max(1, int(len(content) * SAMPLE_RATIO))
                sample_size = min(sample_size, len(content))
                subset = (
                    rng.sample(content, sample_size)
                    if sample_size < len(content)
                    else content
                )
                collected.extend(subset)
                logger.info(
                    f"文件 {json_file.name}: 总样本 {len(content)}，随机提取 {len(subset)} 条"
                )
        except json.JSONDecodeError as exc:
            logger.warning(f"解析 JSON 失败 {json_file.name}: {exc}")
        except Exception as exc:
            logger.error(f"处理文件 {json_file.name} 时出错: {exc}")

    logger.info(f"增强数据提取完成，共收集 {len(collected)} 条样本")
    return collected


def save_enhanced_data(samples, new_data_dir: Path, iteration: int):
    if not samples:
        logger.warning("没有采样到任何增强数据，跳过保存")
        return None, 0

    os.makedirs(new_data_dir, exist_ok=True)
    output_path = new_data_dir / OUTPUT_FILENAME_TEMPLATE.format(iteration=iteration)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info(f"生成了 {len(samples)} 条增强样本，保存到: {output_path}")
    return output_path, len(samples)


def save_summary(
    summary_path: Path, iteration: int, metric: float, sample_count: int, output_path
):
    summary = {
        "iteration": iteration,
        "exact_match_threshold": EXACT_THRESHOLD,
        "exact_match_value": metric,
        "enhancement_triggered": metric < EXACT_THRESHOLD,
        "enhanced_samples": sample_count,
        "summary_file_used": str(summary_path),
        "enhanced_file": str(output_path) if output_path else None,
    }

    summary_output = summary_path.parent / SUMMARY_OUTPUT_TEMPLATE.format(
        iteration=iteration
    )
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"数据增强汇总报告保存到: {summary_output}")


def main():
    if len(sys.argv) < 5:
        usage()
        sys.exit(1)

    config_dir = Path(sys.argv[1])
    new_data_dir = Path(sys.argv[2])
    result_dir = Path(sys.argv[3])
    iteration = int(sys.argv[4])

    logger.info("Plan 数据增强开始:")
    logger.info(f"  配置目录: {config_dir}")
    logger.info(f"  新数据目录: {new_data_dir}")
    logger.info(f"  结果目录: {result_dir}")
    logger.info(f"  迭代次数: {iteration}")

    iteration_dir = resolve_iteration_dir(config_dir)
    enhancement_dir = iteration_dir / "data_enhancement"

    summary_file = find_latest_summary(result_dir)
    if not summary_file:
        logger.warning("未找到 summary_report_*.txt，跳过数据增强")
        return

    logger.info(f"使用汇总报告: {summary_file}")

    exact_match = parse_exact_match(summary_file)
    if exact_match is None:
        logger.warning("无法解析 Overall Exact Match Average，跳过数据增强")
        return

    logger.info(f"总体 Exact Match Average: {exact_match:.4f}")

    if exact_match >= EXACT_THRESHOLD:
        logger.info("Exact Match >= %.2f，本轮无需追加增强数据", EXACT_THRESHOLD)
        return

    samples = collect_random_samples(enhancement_dir)
    if not samples:
        logger.warning("未能收集到任何增强样本")
        return

    output_path, sample_count = save_enhanced_data(samples, new_data_dir, iteration)
    if output_path:
        save_summary(summary_file, iteration, exact_match, sample_count, output_path)

    logger.info("Plan 数据增强完成")


if __name__ == "__main__":
    main()
