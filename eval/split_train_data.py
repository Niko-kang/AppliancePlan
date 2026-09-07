#!/usr/bin/env python3
"""
从 train.json 中切分一定比例的数据作为 test.json
使用固定的 seed 确保每次切分结果一致
"""
import json
import argparse
import random
import os


def split_data(input_file, train_output, test_output, test_ratio=0.05, seed=42):
    """
    从输入文件中切分数据为训练集和测试集

    Args:
        input_file: 输入的 JSON 文件路径
        train_output: 输出的训练集 JSON 文件路径
        test_output: 输出的测试集 JSON 文件路径
        test_ratio: 测试集比例（默认 0.05，即 5%）
        seed: 随机种子（默认 42）
    """
    # 设置随机种子，确保结果可复现
    random.seed(seed)

    # 读取输入文件
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    print(f"正在读取输入文件: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 确保 data 是列表
    if not isinstance(data, list):
        raise ValueError(f"输入文件应包含一个 JSON 数组，但得到类型: {type(data)}")

    total_samples = len(data)
    print(f"总样本数: {total_samples}")

    # 随机打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 计算切分点
    test_size = int(total_samples * test_ratio)
    if test_size == 0:
        test_size = 1
        print(f"警告: 测试集比例过小，至少保留 1 个样本作为测试集")

    train_size = total_samples - test_size
    print(f"训练集样本数: {train_size} ({train_size/total_samples*100:.2f}%)")
    print(f"测试集样本数: {test_size} ({test_size/total_samples*100:.2f}%)")

    # 切分数据
    test_data = shuffled_data[:test_size]
    train_data = shuffled_data[test_size:]

    # 确保输出目录存在
    os.makedirs(
        os.path.dirname(train_output) if os.path.dirname(train_output) else ".",
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(test_output) if os.path.dirname(test_output) else ".",
        exist_ok=True,
    )

    # 保存训练集
    print(f"正在保存训练集到: {train_output}")
    with open(train_output, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # 保存测试集
    print(f"正在保存测试集到: {test_output}")
    with open(test_output, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 数据切分完成")
    print(f"  训练集: {train_output} ({len(train_data)} 个样本)")
    print(f"  测试集: {test_output} ({len(test_data)} 个样本)")


def main():
    parser = argparse.ArgumentParser(
        description="从 train.json 中切分数据为训练集和测试集"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="输入的 train.json 文件路径"
    )
    parser.add_argument(
        "--train_output", type=str, required=True, help="输出的训练集 JSON 文件路径"
    )
    parser.add_argument(
        "--test_output", type=str, required=True, help="输出的测试集 JSON 文件路径"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.05, help="测试集比例（默认 0.05，即 5%）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")

    args = parser.parse_args()

    try:
        split_data(
            args.input_file,
            args.train_output,
            args.test_output,
            args.test_ratio,
            args.seed,
        )
    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
