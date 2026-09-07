#!/usr/bin/env python3
import os
import json
import re
import math
import argparse
from pathlib import Path
from collections import defaultdict
import logging
import time
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# ---------------- 日志 ----------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bbox_eval")

# ---------------- GPU 配置（单机 8 卡）----------------
gpu_list = ["0", "1", "2", "3", "4", "5", "6", "7"]


# ============== 基础工具 ==============


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def reverse_convert_to_original_format(
    bbox_new, orig_height, orig_width, new_height, new_width
):
    """
    从修改后的坐标恢复原始坐标值。

    参数:
    - bbox_new: 修改后的边界框坐标 [x1_new, y1_new, x2_new, y2_new]
    - orig_height: 原始图像的高度
    - orig_width: 原始图像的宽度
    - new_height: 修改后的图像高度
    - new_width: 修改后的图像宽度

    返回:
    - 原始坐标 [x1, y1, x2, y2]
    """
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height

    x1_new, y1_new, x2_new, y2_new = bbox_new

    # 反向计算原始坐标
    x1 = round(x1_new / scale_w)
    y1 = round(y1_new / scale_h)
    x2 = round(x2_new / scale_w)
    y2 = round(y2_new / scale_h)

    # 确保原始坐标在原始图像范围内
    x1 = max(0, min(x1, orig_width - 1))
    y1 = max(0, min(y1, orig_height - 1))
    x2 = max(0, min(x2, orig_width - 1))
    y2 = max(0, min(y2, orig_height - 1))

    # 确保x1 < x2, y1 < y2（如果clip导致反转）
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def reverse_convert_points_to_original(
    points, orig_height, orig_width, new_height, new_width
):
    """
    从修改后的点坐标恢复原始坐标值

    参数:
    - points: 修改后的点列表 [[x1, y1], [x2, y2], ...]
    - orig_height: 原始图像的高度
    - orig_width: 原始图像的宽度
    - new_height: 修改后的图像高度
    - new_width: 修改后的图像宽度

    返回:
    - 原始点坐标列表 [[x1, y1], [x2, y2], ...]
    """
    if not points:
        return []

    scale_w = new_width / orig_width
    scale_h = new_height / orig_height

    original_points = []
    for point in points:
        x_new, y_new = point
        x = round(x_new / scale_w)
        y = round(y_new / scale_h)
        # 确保坐标在原始图像范围内
        x = max(0, min(x, orig_width - 1))
        y = max(0, min(y, orig_height - 1))
        original_points.append([x, y])

    return original_points


def process_images(img_paths):
    """处理图像并返回处理后的图像列表和尺寸信息（保持原图尺寸）"""
    from PIL import ImageOps

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


def parse_coordinates_from_text(text):
    """
    从文本中解析坐标数据，支持多种格式：
    - points: [(x1, y1), (x2, y2), ...]
    - bbox: [(x1, y1, x2, y2), ...]
    - JSON bbox: {"bbox_2d": [x1, y1, x2, y2]}
    返回: (coordinates, coord_type) 或 (None, None)
    """
    coordinates = []
    coord_type = None  # 'points' or 'bbox'

    # 尝试解析 JSON 格式的 bbox
    json_bbox_match = re.search(r'\{"bbox_2d":\s*\[([^\]]+)\]\}', text)
    if json_bbox_match:
        coords_str = json_bbox_match.group(1)
        try:
            coords = [float(x.strip()) for x in coords_str.split(",")]
            if len(coords) == 4:
                coordinates.append(coords)
                coord_type = "bbox"
                return coordinates, coord_type
        except Exception:
            pass

    # 尝试解析列表格式 [(x1, y1, x2, y2), ...] 或 [(x1, y1), ...]
    full_list_match = re.search(r"\[((?:\([^)]+\))(?:,\s*\([^)]+\))*)\]", text)
    if full_list_match:
        content = full_list_match.group(1)
        tuple_pattern = r"\(([^)]+)\)"
        tuple_matches = re.findall(tuple_pattern, content)

        for tuple_match in tuple_matches:
            try:
                coords = [float(x.strip()) for x in tuple_match.split(",")]
                if len(coords) == 2:
                    coordinates.append(coords)
                    if coord_type is None:
                        coord_type = "points"
                elif len(coords) == 4:
                    coordinates.append(coords)
                    if coord_type is None:
                        coord_type = "bbox"
            except (ValueError, AttributeError):
                continue

        if coordinates:
            return coordinates, coord_type

    # 如果没有匹配到完整列表，尝试匹配单个坐标列表 [x1, y1, x2, y2] 或 [x1, y1]
    single_list_pattern = r"\[([\d.,\s-]+)\]"
    single_matches = re.findall(single_list_pattern, text)

    for match in single_matches:
        try:
            coords = [float(x.strip()) for x in match.split(",")]
            if len(coords) == 2:
                coordinates.append(coords)
                if coord_type is None:
                    coord_type = "points"
            elif len(coords) == 4:
                coordinates.append(coords)
                if coord_type is None:
                    coord_type = "bbox"
        except ValueError:
            continue

    # 尝试匹配 "bbox_2d: [x1, y1, x2, y2]" 格式
    m = re.search(r"bbox_2d\s*[:=]\s*\[([^\]]+)\]", text)
    if m:
        try:
            nums = [float(x.strip()) for x in m.group(1).split(",")]
            if len(nums) == 4:
                return [nums], "bbox"
        except Exception:
            pass

    if coordinates:
        return coordinates, coord_type
    return None, None


def extract_bbox(text):
    """
    从模型文本输出中提取 bbox_2d（向后兼容）
    期望格式：...{"bbox_2d":[x1,y1,x2,y2]}...
    返回：list 或 None
    """
    coords, coord_type = parse_coordinates_from_text(text)
    if coord_type == "bbox" and coords and len(coords) > 0:
        # 如果只有一个bbox，返回第一个；否则返回None（表示格式不支持）
        if len(coords) == 1:
            return coords[0]
    return None


def scale_pred_to_original(pred_bbox, orig_w, orig_h):
    """
    将预测框从图片实际尺寸映射到原图坐标：
    假设模型输出的bbox是基于图片实际尺寸的像素坐标
    返回整数像素 [x1,y1,x2,y2]（裁剪边界）
    """
    if pred_bbox is None:
        return None

    x1, y1, x2, y2 = [float(v) for v in pred_bbox]

    # 直接使用原图尺寸，不需要缩放
    X1 = int(round(x1))
    Y1 = int(round(y1))
    X2 = int(round(x2))
    Y2 = int(round(y2))

    # 规范次序与边界裁剪
    x1o, x2o = sorted((X1, X2))
    y1o, y2o = sorted((Y1, Y2))
    x1o = max(0, min(x1o, orig_w - 1))
    y1o = max(0, min(y1o, orig_h - 1))
    x2o = max(0, min(x2o, orig_w - 1))
    y2o = max(0, min(y2o, orig_h - 1))
    if x2o <= x1o or y2o <= y1o:
        return None
    return [x1o, y1o, x2o, y2o]


def calculate_iou(b1, b2):
    """计算两个 bbox 的 IoU（均为 [x1,y1,x2,y2]，像素坐标）"""
    if not b1 or not b2:
        return 0.0
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def center_distance(boxA, boxB):
    """计算两个bbox中心点的距离"""
    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2
    return ((cxA - cxB) ** 2 + (cyA - cyB) ** 2) ** 0.5


def point_distance(p1, p2):
    """计算两个点之间的欧氏距离"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def calculate_mask_accuracy(pred_points, mask_path, orig_height=None, orig_width=None):
    """
    计算预测点在mask内的准确率
    Args:
        pred_points: 预测的点列表 [[x1, y1], [x2, y2], ...]（绝对坐标）
        mask_path: mask图片路径
        orig_height: 原始图片高度（用于归一化坐标转换）
        orig_width: 原始图片宽度（用于归一化坐标转换）
    Returns:
        float: 准确率（0.0-1.0），如果失败返回None
    """
    if not pred_points or not mask_path:
        return None

    if not os.path.exists(mask_path):
        logger.warning(f"Mask file not found: {mask_path}")
        return None

    try:
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img) / 255.0

        # 如果是RGB图片，取第一个通道
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        mask_height, mask_width = mask.shape

        # 判断坐标是否为归一化坐标（如果提供了orig_height和orig_width）
        is_normalized = False
        if orig_height and orig_width:
            # 检查是否所有点都在[0,1]范围内
            is_normalized = all(0 <= x <= 1 and 0 <= y <= 1 for x, y in pred_points)

        correct_count = 0

        for x, y in pred_points:
            # 如果是归一化坐标，转换为绝对坐标
            if is_normalized and orig_height and orig_width:
                x = int(round(x * orig_width))
                y = int(round(y * orig_height))
            else:
                x = int(round(x))
                y = int(round(y))

            # 检查坐标是否在mask范围内
            if 0 <= x < mask_width and 0 <= y < mask_height:
                if mask[y, x] > 0.5:  # mask值>0.5认为是前景
                    correct_count += 1

        accuracy = correct_count / len(pred_points) if pred_points else 0.0
        return accuracy

    except Exception as e:
        logger.warning(f"Error calculating mask accuracy for {mask_path}: {e}")
        return None


def evaluate_points(
    pred_points,
    gt_points,
    threshold=5.0,
    mask_path=None,
    orig_height=None,
    orig_width=None,
):
    """
    评估点坐标预测
    Args:
        pred_points: 预测的点列表 [[x1, y1], [x2, y2], ...]
        gt_points: 真实点列表 [[x1, y1], [x2, y2], ...]
        threshold: 判断点是否正确的距离阈值（像素）
        mask_path: mask图片路径（可选，用于mask评估）
        orig_height: 原始图片高度（用于mask评估时的归一化坐标转换）
        orig_width: 原始图片宽度（用于mask评估时的归一化坐标转换）
    Returns:
        dict: 包含评估指标的字典
    """
    result = {
        "avg_distance": None,
        "min_distance": None,
        "hit_rate": 0.0,
        "num_pred": len(pred_points) if pred_points else 0,
        "num_gt": len(gt_points) if gt_points else 0,
        "mask_accuracy": None,
    }

    if not pred_points or not gt_points:
        return result

    # 计算每个预测点到最近GT点的距离
    distances = []
    for pred_p in pred_points:
        min_dist = float("inf")
        for gt_p in gt_points:
            dist = point_distance(pred_p, gt_p)
            min_dist = min(min_dist, dist)
        distances.append(min_dist)

    # 计算每个GT点到最近预测点的距离（用于hit_rate）
    gt_hits = []
    for gt_p in gt_points:
        min_dist = float("inf")
        for pred_p in pred_points:
            dist = point_distance(pred_p, gt_p)
            min_dist = min(min_dist, dist)
        gt_hits.append(min_dist <= threshold)

    avg_distance = sum(distances) / len(distances) if distances else None
    min_distance = min(distances) if distances else None
    hit_rate = sum(gt_hits) / len(gt_hits) if gt_hits else 0.0

    result.update(
        {
            "avg_distance": avg_distance,
            "min_distance": min_distance,
            "hit_rate": hit_rate,
            "all_distances": distances,
        }
    )

    # 如果有mask路径，计算mask准确率
    if mask_path:
        mask_acc = calculate_mask_accuracy(
            pred_points, mask_path, orig_height, orig_width
        )
        result["mask_accuracy"] = mask_acc

    return result


# def draw_and_save(img_path, pred_bbox, gt_bbox, iou, out_dir, base_name):
#     """将预测与GT绘制在原图上并保存 - 参考visualize_bbox_results"""
#     os.makedirs(out_dir, exist_ok=True)

#     try:
#         img = Image.open(img_path).convert("RGB")
#     except Exception as e:
#         logger.warning(f"无法加载图像 {img_path}: {e}")
#         return

#     fig, ax = plt.subplots(1, figsize=(8, 8))
#     ax.imshow(img)

#     # GT: 绿色矩形
#     if gt_bbox:
#         x1, y1, x2, y2 = gt_bbox
#         ax.add_patch(
#             plt.Rectangle(
#                 (x1, y1),
#                 x2 - x1,
#                 y2 - y1,
#                 linewidth=2,
#                 edgecolor="g",
#                 facecolor="none",
#                 label="GT",
#             )
#         )

#     # Pred: 红色矩形
#     if pred_bbox:
#         x1, y1, x2, y2 = pred_bbox
#         ax.add_patch(
#             plt.Rectangle(
#                 (x1, y1),
#                 x2 - x1,
#                 y2 - y1,
#                 linewidth=2,
#                 edgecolor="r",
#                 facecolor="none",
#                 label=f"Pred (IoU={iou:.3f})" if iou is not None else "Pred",
#             )
#         )

#     ax.axis("off")
#     ax.legend()

#     out_path = os.path.join(out_dir, f"{base_name}.png")
#     plt.savefig(out_path, bbox_inches="tight")
#     plt.close(fig)


def draw_and_save(
    img_path, pred_coords, gt_coords, coord_type, metrics, out_dir, base_name
):
    """将预测与GT绘制在原图上并保存 - 支持bbox和points"""
    os.makedirs(out_dir, exist_ok=True)

    try:
        # 使用 PIL 读取并按 EXIF 方向矫正
        from PIL import ImageOps

        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        logger.warning(f"无法加载图像 {img_path}: {e}")
        return

    # 创建绘图对象
    draw = ImageDraw.Draw(img)
    w, h = img.size

    def _draw_box(bbox, color, text=None):
        """绘制单个bbox框"""
        if not bbox:
            return
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        # 确保 x1 < x2, y1 < y2
        if x2 <= x1 or y2 <= y1:
            return
        # 绘制矩形（PIL的rectangle使用 (x1, y1, x2, y2) 格式）
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # 绘制文本标签
        if text:
            ty = max(0, y1 - 15)
            draw.text((x1, ty), text, fill=color)

    def _draw_points(points, color, text=None, radius=3):
        """绘制点列表"""
        if not points:
            return
        for i, point in enumerate(points):
            x, y = [int(round(v)) for v in point]
            # 确保坐标在图像范围内
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            # 绘制圆点
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color,
                outline=color,
            )
            # 如果是第一个点，绘制标签
            if text and i == 0:
                ty = max(0, y - 15)
                draw.text((x, ty), text, fill=color)

    if coord_type == "bbox":
        # 绘制 GT（绿色）与 Pred（红色）
        _draw_box(gt_coords[0] if gt_coords else None, (0, 255, 0), "GT")
        pred_label = (
            f"Pred (IoU={metrics.get('iou', 0):.3f})"
            if metrics.get("iou") is not None
            else "Pred"
        )
        _draw_box(pred_coords[0] if pred_coords else None, (255, 0, 0), pred_label)
    elif coord_type == "points":
        # 绘制点
        _draw_points(gt_coords, (0, 255, 0), "GT", radius=4)
        hit_rate = metrics.get("hit_rate", 0)
        avg_dist = metrics.get("avg_distance", 0)
        if avg_dist is not None:
            pred_label = f"Pred (HR={hit_rate:.2f}, AvgDist={avg_dist:.1f})"
        else:
            pred_label = "Pred"
        _draw_points(pred_coords, (255, 0, 0), pred_label, radius=3)

    # 保存图像（PIL直接保存RGB格式）
    out_path = os.path.join(out_dir, f"{base_name}.png")
    img.save(out_path)


# ============== 模型相关 ==============


def load_model(model_name, gpu_id, root_path):
    """加载模型与处理器（输入图片保持原尺寸）"""
    if os.path.isabs(model_name):
        model_path = model_name
    else:
        model_path = f"{root_path}ft/scripts/Model_ftd/{model_name}"

    logger.info(f"Loading model from: {model_path} on GPU {gpu_id}")
    processor = AutoProcessor.from_pretrained(model_path)

    llm = LLM(
        model=model_path, tensor_parallel_size=1, limit_mm_per_prompt={"image": 25}
    )
    return llm, processor


def run_bbox_inference(
    llm,
    processor,
    test_data,
    output_file,
    max_new_tokens=512,
    visualize_dir=None,
    gpu_id=0,
):
    """
    核心推理：参考run_bbox函数的逻辑
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        max_tokens=max_new_tokens,
        stop_token_ids=[processor.tokenizer.eos_token_id],
    )

    results = []
    total = 0
    # 统计信息
    bbox_stats = {"valid": 0, "iou_sum": 0.0, "dist_sum": 0.0, "above_05": 0}
    points_stats = {
        "valid": 0,
        "dist_sum": 0.0,
        "hit_sum": 0.0,
        "mask_accuracy_sum": 0.0,
        "mask_valid": 0,
    }

    sample_id = 0
    for i, sample in enumerate(tqdm(test_data, desc="Coordinate Inference")):
        try:
            question = sample["conversations"][0]["value"]

            # 解析GT坐标（支持多种格式）
            gt_text = sample["conversations"][1]["value"]
            gt_coords, gt_coord_type = parse_coordinates_from_text(gt_text)

            # 如果解析失败，尝试旧的JSON格式
            if not gt_coords:
                try:
                    gt_json = json.loads(gt_text)
                    if isinstance(gt_json, dict) and "bbox_2d" in gt_json:
                        gt_coords = [gt_json["bbox_2d"]]
                        gt_coord_type = "bbox"
                except Exception:
                    pass

            img_paths = sample.get("image", [])
            if isinstance(img_paths, str):
                img_paths = [img_paths]
            images, _ = process_images(img_paths)

            # 获取mask路径（如果存在）
            mask_path = None
            if "mask" in sample:
                mask_rel_path = sample["mask"]
                # 尝试从图片路径推断mask目录
                if img_paths:
                    img_dir = os.path.dirname(img_paths[0])
                    # 如果图片路径包含images，替换为masks
                    if "images" in img_dir:
                        mask_dir = img_dir.replace("images", "masks")
                    else:
                        # 否则使用图片目录的父目录下的masks
                        mask_dir = os.path.join(os.path.dirname(img_dir), "masks")
                    mask_path = os.path.join(mask_dir, mask_rel_path)
                    # 如果不存在，尝试绝对路径
                    if not os.path.exists(mask_path):
                        # 尝试使用ROOT路径
                        if "/pretrain_data/RoboAfford-Eval/" in img_paths[0]:
                            mask_path = (
                                img_paths[0]
                                .replace(
                                    "/pretrain_data/RoboAfford-Eval/images/",
                                    "/pretrain_data/RoboAfford-Eval/masks/",
                                )
                                .replace(os.path.basename(img_paths[0]), mask_rel_path)
                            )
                        else:
                            # 最后尝试相对路径
                            base_dir = os.path.dirname(os.path.dirname(img_paths[0]))
                            mask_path = os.path.join(base_dir, "masks", mask_rel_path)
            elif "mask_path" in sample:
                mask_path = sample["mask_path"]

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}]
                    + [{"image": img} for img in images],
                }
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            multi_modal_data = {"image": [img for img in images]}
            outputs = llm.generate(
                [{"prompt": prompt, "multi_modal_data": multi_modal_data}],
                sampling_params=sampling_params,
            )
            pred_text = outputs[0].outputs[0].text.strip()

            # 解析预测坐标（支持多种格式）
            pred_coords, pred_coord_type = parse_coordinates_from_text(pred_text)

            # 获取图片尺寸
            ori_image = cv2.imread(img_paths[-1] if img_paths else img_paths[0])
            orig_height, orig_width = ori_image.shape[:2]
            new_height, new_width = smart_resize(orig_height, orig_width)

            # 转换坐标到原图尺寸
            metrics = {}
            coord_type = gt_coord_type or pred_coord_type
            if pred_coords and gt_coords and coord_type:
                # 确保预测和GT的类型匹配
                if (
                    pred_coord_type
                    and gt_coord_type
                    and pred_coord_type != gt_coord_type
                ):
                    logger.warning(
                        f"Sample {i}: Coord type mismatch (pred: {pred_coord_type}, gt: {gt_coord_type})"
                    )
                    pred_coords = None
                    gt_coords = None
                else:
                    if pred_coord_type == "bbox":
                        # 转换bbox坐标
                        pred_bbox = reverse_convert_to_original_format(
                            pred_coords[0],
                            orig_height,
                            orig_width,
                            new_height,
                            new_width,
                        )
                        gt_bbox = reverse_convert_to_original_format(
                            gt_coords[0], orig_height, orig_width, new_height, new_width
                        )
                        if pred_bbox and gt_bbox:
                            iou = calculate_iou(pred_bbox, gt_bbox)
                            dist = center_distance(gt_bbox, pred_bbox)
                            metrics = {
                                "iou": iou,
                                "center_distance": dist,
                                "type": "bbox",
                            }
                            bbox_stats["valid"] += 1
                            bbox_stats["iou_sum"] += iou
                            bbox_stats["dist_sum"] += dist
                            if iou >= 0.5:
                                bbox_stats["above_05"] += 1
                            pred_coords = [pred_bbox]
                            gt_coords = [gt_bbox]
                        else:
                            pred_coords = None
                            gt_coords = None
                    elif pred_coord_type == "points":
                        # 转换点坐标
                        pred_points = reverse_convert_points_to_original(
                            pred_coords, orig_height, orig_width, new_height, new_width
                        )
                        gt_points = reverse_convert_points_to_original(
                            gt_coords, orig_height, orig_width, new_height, new_width
                        )
                        if pred_points and gt_points:
                            # 评估点（包括mask评估，如果有mask路径）
                            point_metrics = evaluate_points(
                                pred_points,
                                gt_points,
                                mask_path=mask_path,
                                orig_height=orig_height,
                                orig_width=orig_width,
                            )
                            metrics = {**point_metrics, "type": "points"}
                            points_stats["valid"] += 1
                            if point_metrics["avg_distance"] is not None:
                                points_stats["dist_sum"] += point_metrics[
                                    "avg_distance"
                                ]
                                points_stats["hit_sum"] += point_metrics["hit_rate"]
                            # 如果有mask准确率，也统计
                            if point_metrics.get("mask_accuracy") is not None:
                                points_stats["mask_accuracy_sum"] += point_metrics[
                                    "mask_accuracy"
                                ]
                                points_stats["mask_valid"] += 1
                            pred_coords = pred_points
                            gt_coords = gt_points
                        else:
                            pred_coords = None
                            gt_coords = None

            total += 1

            res = {
                "id": sample_id,
                "question": question,
                "answer_raw": pred_text,
                "coord_type": gt_coord_type,
                "pred_coords": pred_coords,
                "gt_coords": gt_coords,
                "metrics": metrics,
                "image_paths": img_paths,
                "mask_path": mask_path,
                "orig_size": [orig_width, orig_height],
            }
            results.append(res)

            # 可视化
            if visualize_dir:
                base_name = f"{Path(img_paths[-1] if img_paths else 'unknown').stem}_{sample_id}_{gpu_id}"
                draw_and_save(
                    img_paths[-1] if img_paths else img_paths[0],
                    pred_coords,
                    gt_coords,
                    coord_type,
                    metrics,
                    out_dir=os.path.join(visualize_dir, "images"),
                    base_name=base_name,
                )
            sample_id += 1

        except Exception as e:
            logger.exception(f"Sample {i} failed: {e}")

    # 写结果
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 写汇总
    txt_file = output_file.replace(".jsonl", ".txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"Total samples: {total}\n\n")

        # BBox统计
        f.write("=== BBox Evaluation ===\n")
        f.write(f"Valid bbox pairs: {bbox_stats['valid']}\n")
        if bbox_stats["valid"] > 0:
            f.write(f"Average IoU: {bbox_stats['iou_sum']/bbox_stats['valid']:.4f}\n")
            f.write(
                f"Average center distance: {bbox_stats['dist_sum']/bbox_stats['valid']:.2f} px\n"
            )
            f.write(
                f"IoU >= 0.5 ratio: {bbox_stats['above_05']}/{bbox_stats['valid']} ({bbox_stats['above_05']/bbox_stats['valid']:.2%})\n"
            )
        else:
            f.write("No valid bbox pairs\n")

        f.write("\n=== Points Evaluation ===\n")
        f.write(f"Valid points pairs: {points_stats['valid']}\n")
        if points_stats["valid"] > 0:
            f.write(
                f"Average point distance: {points_stats['dist_sum']/points_stats['valid']:.2f} px\n"
            )
            f.write(
                f"Average hit rate: {points_stats['hit_sum']/points_stats['valid']:.2%}\n"
            )
            # Mask准确率统计
            if points_stats.get("mask_valid", 0) > 0:
                mask_acc = (
                    points_stats["mask_accuracy_sum"] / points_stats["mask_valid"]
                )
                f.write(
                    f"Mask accuracy: {mask_acc:.4f} ({points_stats['mask_valid']}/{points_stats['valid']} samples with mask)\n"
                )
        else:
            f.write("No valid points pairs\n")

    logger.info(f"Saved results to {output_file}")
    if bbox_stats["valid"] > 0:
        logger.info(
            f"BBox - Average IoU: {bbox_stats['iou_sum']/bbox_stats['valid']:.4f}"
        )
    if points_stats["valid"] > 0:
        logger.info(
            f"Points - Average distance: {points_stats['dist_sum']/points_stats['valid']:.2f} px, Hit rate: {points_stats['hit_sum']/points_stats['valid']:.2%}"
        )
        if points_stats.get("mask_valid", 0) > 0:
            mask_acc = points_stats["mask_accuracy_sum"] / points_stats["mask_valid"]
            logger.info(
                f"Points - Mask accuracy: {mask_acc:.4f} ({points_stats['mask_valid']}/{points_stats['valid']} samples)"
            )

    return results


# ============== 多进程：单卡单实例处理"多文件" ==============


def process_file_list(
    model_name,
    json_file_list,
    gpu_id,
    root_path,
    custom_result_dir,
):
    """每张GPU一个进程：加载一次模型，顺序处理分配的文件"""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(
            f"[GPU {gpu_id}] Loading model ONCE for {len(json_file_list)} files: {model_name}"
        )
        llm, processor = load_model(model_name, gpu_id, root_path)
        logger.info(f"[GPU {gpu_id}] Model loaded.")

        all_results = []
        data_dir = None

        for jf in json_file_list:
            try:
                p = Path(jf)
                with p.open("r", encoding="utf-8") as f:
                    if p.suffix == ".json":
                        test_data = json.load(f)
                        if not isinstance(test_data, list):
                            test_data = [test_data]
                    else:
                        test_data = [json.loads(line) for line in f]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_file = os.path.join(
                    custom_result_dir, f"bbox_{p.stem}_inference_{timestamp}.jsonl"
                )

                logger.info(
                    f"[GPU {gpu_id}] Processing: {p.name} ({len(test_data)} samples)"
                )
                res = run_bbox_inference(
                    llm,
                    processor,
                    test_data,
                    out_file,
                    visualize_dir=custom_result_dir,
                    gpu_id=gpu_id,
                )
                all_results.extend(res)
                data_dir = p.parent

                logger.info(f"✓ [GPU {gpu_id}] {p.name}: {len(res)} samples processed")

            except Exception as e:
                logger.exception(f"[GPU {gpu_id}] Error processing {jf}: {e}")

        logger.info(f"✅ [GPU {gpu_id}] Finished {len(json_file_list)} files.")
        return (model_name, all_results, data_dir)

    except Exception as e:
        logger.exception(f"❌ [GPU {gpu_id}] Error in process_file_list: {e}")
        return (model_name, [], None)


# ============== 主函数 ==============


def main():
    parser = argparse.ArgumentParser(
        description="BBox Evaluation Script - Single Node 8 GPU (orig-size input)"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model path or name"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Test data directory"
    )
    parser.add_argument("--root_path", type=str, required=True, help="Root path")
    parser.add_argument(
        "--result_dir", type=str, required=True, help="Result directory"
    )
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    logger.info(
        "Starting Coordinate evaluation (BBox + Points) - Single Node 8 GPU, 1 instance/GPU"
    )
    logger.info(f"GPUs: {gpu_list} (count={len(gpu_list)})")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Result dir: {args.result_dir}")
    logger.info("Prediction space: using image actual size")

    start = time.time()

    # 收集数据文件
    data_path = Path(args.data_path)
    json_files = sorted(
        list(data_path.glob("*.json")) + list(data_path.glob("*.jsonl"))
    )
    if not json_files:
        logger.warning(f"No JSON/JSONL files found in {args.data_path}")
        return
    logger.info(f"Found {len(json_files)} files.")

    # 轮询分配给 8 张 GPU
    max_workers = len(gpu_list)
    gpu_file_lists = [[] for _ in range(max_workers)]
    for idx, p in enumerate(json_files):
        gpu_file_lists[idx % max_workers].append(str(p))

    for i, flist in enumerate(gpu_file_lists):
        if flist:
            logger.info(f"GPU {gpu_list[i]}: {len(flist)} files")

    # 并行执行
    all_results_by_model = defaultdict(dict)
    completed = 0
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i in range(max_workers):
            if not gpu_file_lists[i]:
                continue
            futures.append(
                ex.submit(
                    process_file_list,
                    args.model_name,
                    gpu_file_lists[i],
                    gpu_list[i],
                    args.root_path,
                    args.result_dir,
                )
            )

        total = len(futures)
        for fut in as_completed(futures):
            completed += 1
            try:
                ret = fut.result()
                if ret and len(ret) == 3:
                    mname, res, ddir = ret
                    if mname not in all_results_by_model:
                        all_results_by_model[mname] = {"results": [], "data_dir": ddir}
                    all_results_by_model[mname]["results"].extend(res)
                    if ddir:
                        all_results_by_model[mname]["data_dir"] = ddir
                logger.info(f"✅ [{completed}/{total}] GPU finished.")
            except Exception as e:
                logger.error(f"❌ [{completed}/{total}] GPU failed: {e}")

    # 汇总日志
    logger.info("\n" + "=" * 80)
    logger.info("Final evaluation summary:")
    if args.model_name in all_results_by_model:
        results = all_results_by_model[args.model_name].get("results", [])
        if results:
            # 按类型分组统计
            bbox_results = [
                r for r in results if r.get("metrics", {}).get("type") == "bbox"
            ]
            points_results = [
                r for r in results if r.get("metrics", {}).get("type") == "points"
            ]

            if bbox_results:
                valid_bbox = [
                    r
                    for r in bbox_results
                    if r.get("metrics", {}).get("iou") is not None
                ]
                if valid_bbox:
                    avg_iou = sum(r["metrics"]["iou"] for r in valid_bbox) / len(
                        valid_bbox
                    )
                    above_05 = sum(1 for r in valid_bbox if r["metrics"]["iou"] >= 0.5)
                    logger.info(f"BBox - Average IoU: {avg_iou:.4f}")
                    logger.info(
                        f"BBox - IoU >= 0.5 ratio: {above_05}/{len(valid_bbox)} ({above_05/len(valid_bbox):.2%})"
                    )
                else:
                    logger.info("BBox - No valid bbox pairs in results.")

            if points_results:
                valid_points = [
                    r
                    for r in points_results
                    if r.get("metrics", {}).get("avg_distance") is not None
                ]
                if valid_points:
                    avg_dist = sum(
                        r["metrics"]["avg_distance"] for r in valid_points
                    ) / len(valid_points)
                    avg_hit = sum(r["metrics"]["hit_rate"] for r in valid_points) / len(
                        valid_points
                    )
                    logger.info(f"Points - Average distance: {avg_dist:.2f} px")
                    logger.info(f"Points - Average hit rate: {avg_hit:.2%}")
                else:
                    logger.info("Points - No valid points pairs in results.")

            if not bbox_results and not points_results:
                logger.info("No valid coordinate results found.")
        else:
            logger.info("No results collected.")
    else:
        logger.info("No results for the model key.")

    dur = time.time() - start
    logger.info("\n" + "=" * 80)
    logger.info(f"All tasks completed in {dur:.2f}s ({dur/60:.2f} min)")
    logger.info(f"Total files processed: {len(json_files)}")
    logger.info("=" * 80)
    logger.info("BBox evaluation completed!")


if __name__ == "__main__":
    main()
