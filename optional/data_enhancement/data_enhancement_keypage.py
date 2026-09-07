#!/usr/bin/env python3
"""
Keypage数据增强脚本：根据测评结果生成新的训练数据
逻辑：
1. 读取 keypage_summary_report_*.txt
2. 若 Recall 或 F1 Score 任意一个 < 0.7，则从增强目录中每个 JSON 随机抽取 1/4 数据用于增强
"""
import os
import sys
import json
from pathlib import Path
import logging
import random
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUMMARY_PATTERN = re.compile(r"keypage_summary_report.*\\.txt$")


def extract_image_filename(image_path):
    """从图片路径中提取文件名"""
    if isinstance(image_path, list):
        if not image_path:
            return None
        image_path = image_path[-1]

    if isinstance(image_path, str):
        return os.path.basename(image_path)

    return None


def extract_metrics_from_summary(summary_file: Path):
    """从 summary_report 文件中解析 Recall 和 F1 Score"""
    recall = None
    f1 = None

    with summary_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Recall"):
                try:
                    recall = float(line.split(":")[1])
                except ValueError:
                    logger.warning(f"解析 Recall 失败: {line}")
            elif line.startswith("F1 Score"):
                try:
                    f1 = float(line.split(":")[1])
                except ValueError:
                    logger.warning(f"解析 F1 Score 失败: {line}")

    return recall, f1


def collect_random_quarter(enhancement_dir):
    """从增强目录中每个 JSON 文件随机提取 1/4 样本"""
    collected = []

    if not os.path.exists(enhancement_dir):
        logger.warning(f"数据增强目录不存在: {enhancement_dir}")
        return collected

    json_files = [
        p
        for p in Path(enhancement_dir).rglob("*.json")
        if p.is_file() and not p.name.endswith("checkpoint.json")
    ]

    if not json_files:
        logger.warning(f"在 {enhancement_dir} 中未找到 JSON 文件")
        return collected

    rng = random.Random(42)

    logger.info(f"准备从 {len(json_files)} 个增强文件中随机提取 1/4 样本")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if not isinstance(content, list):
                    content = [content]
                if not content:
                    continue
                quarter = max(1, len(content) // 2)
                subset = (
                    rng.sample(content, quarter) if quarter < len(content) else content
                )
                collected.extend(subset)
                logger.info(
                    f"文件 {json_file.name}: 总样本 {len(content)}，随机提取 {len(subset)} 条"
                )
        except json.JSONDecodeError as e:
            logger.warning(f"解析 JSON 文件失败 {json_file.name}: {e}")
        except Exception as e:
            logger.error(f"处理文件 {json_file.name} 时出错: {e}")

    logger.info(f"增强数据提取完成，共收集 {len(collected)} 条样本")
    return collected


def process_error_results(result_dir, iteration_dir):
    """
    检查 summary_report，决定是否需要追加增强数据
    """
    enhancement_dir = os.path.join(iteration_dir, "data_enhancement_keypage")

    logger.info(f"开始处理测评结果目录: {result_dir}")

    result_path = Path(result_dir)
    print(result_path)
    # summary_files = [
    #     p for p in result_path.rglob("*.txt") if SUMMARY_PATTERN.match(p.name)
    # ]
    summary_files = list(result_path.glob("keypage_summary_report*.txt"))

    if not summary_files:
        logger.warning("未找到 keypage 汇总报告，跳过数据增强")
        return []

    # 使用最新的 summary 文件
    summary_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    summary_file = summary_files[0]
    logger.info(f"使用汇总报告: {summary_file}")

    recall, f1 = extract_metrics_from_summary(summary_file)

    if recall is None or f1 is None:
        logger.warning("汇总报告中缺少 Recall 或 F1 Score，跳过数据增强")
        return []

    logger.info(f"当前 Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    if recall >= 0.7 and f1 >= 0.7:
        logger.info("Recall 和 F1 Score 均 ≥ 0.7，本轮无需追加增强数据")
        return []

    logger.info("性能低于阈值，开始提取随机增强数据")
    return collect_random_quarter(enhancement_dir)


def main():
    """主函数"""
    if len(sys.argv) < 5:
        print(
            "用法: python data_enhancement_keypage.py <config_dir> <new_data_dir> <result_dir> <iteration>"
        )
        print("参数说明:")
        print("  config_dir: 配置目录的绝对路径（通常为 ITERATION_DIR/log）")
        print("  new_data_dir: 新训练数据目录的绝对路径")
        print("  result_dir: 测评结果目录的绝对路径（包含所有 *.jsonl 文件）")
        print("  iteration: 当前迭代次数（用于确定 ITERATION_DIR）")
        sys.exit(1)

    config_dir = sys.argv[1]
    new_data_dir = sys.argv[2]
    result_dir = sys.argv[3]
    iteration = int(sys.argv[4])

    logger.info(f"Keypage数据增强开始:")
    logger.info(f"  配置目录: {config_dir}")
    logger.info(f"  新数据目录: {new_data_dir}")
    logger.info(f"  结果目录: {result_dir}")
    logger.info(f"  迭代次数: {iteration}")

    # 确保目录存在
    os.makedirs(new_data_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # 确定迭代目录
    # config_dir 传入的是 ${ITERATION_DIR}/log，需要向上找到 ITERATION_DIR
    # 数据增强文件夹在 ${ITERATION_DIR}/data_enhancement_keypage/
    if config_dir.endswith("/log"):
        iteration_dir = os.path.dirname(config_dir)
    else:
        iteration_dir = config_dir  # 如果路径不是以 /log 结尾，使用原路径
    logger.info(f"迭代目录: {iteration_dir}")

    # 处理测评结果，提取错误样本并查找匹配的完整数据
    collected_data = process_error_results(result_dir, iteration_dir)

    if not collected_data:
        logger.warning("未找到任何匹配的数据，跳过数据增强")
        return

    logger.info(f"总共收集到 {len(collected_data)} 条数据，开始保存...")

    if collected_data:
        # 保存增强数据到不同的文件名中（按任务类型）
        output_file = os.path.join(
            new_data_dir, f"enhanced_keypage_iter_{iteration}.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(collected_data, f, ensure_ascii=False, indent=2)

        logger.info(f"生成了 {len(collected_data)} 个增强样本，保存到: {output_file}")

        # 生成汇总报告
        summary = {
            "iteration": iteration,
            "total_collected_samples": len(collected_data),
            "source_dir": result_dir,
            "enhancement_dir": os.path.join(iteration_dir, "data_enhancement_keypage"),
            "output_file": output_file,
        }

        summary_file = os.path.join(
            config_dir, f"enhancement_keypage_summary_iter_{iteration}.json"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Keypage数据增强汇总报告保存到: {summary_file}")

        print(f"\nKeypage数据增强统计:")
        print(f"  收集到的数据: {len(collected_data)}")
        print(f"  输出文件: {output_file}")

    else:
        logger.warning("未收集到任何数据")

    logger.info("Keypage数据增强完成")


if __name__ == "__main__":
    main()
