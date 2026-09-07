import re
from difflib import SequenceMatcher


def find_error_steps(gt_text, pred_text):
    """
    比较预测结果与标准答案，返回错误步骤的编号列表

    Args:
        gt_text (str): 标准答案的操作步骤文本
        pred_text (str): 预测的操作步骤文本

    Returns:
        list: 错误步骤的编号列表（从1开始计数）
    """

    def parse_steps(text):
        """将文本解析为步骤列表"""
        steps = [
            line.strip().lower() for line in text.strip().split("\n") if line.strip()
        ]
        return steps

    def extract_action_and_params(step):
        """提取动作名与参数（例如 rotate, 定时旋钮, 50min, 顺时针旋转200°）"""
        match = re.match(r"(\w+)\((.*)\)", step)
        if not match:
            return step, []
        action = match.group(1)
        params = [p.strip() for p in match.group(2).split(",")]
        return action, params

    def compare_plans(gt_text, pred_text):
        gt_steps = parse_steps(gt_text)
        pred_steps = parse_steps(pred_text)

        # 用 difflib 找出最长匹配序列
        sm = SequenceMatcher(None, gt_steps, pred_steps)
        opcodes = sm.get_opcodes()

        errors = []

        for tag, i1, i2, j1, j2 in opcodes:
            # tag 可以是 'equal', 'replace', 'delete', 'insert'
            if tag == "equal":
                # 检查参数是否一致
                for k in range(i2 - i1):
                    gt_action, gt_params = extract_action_and_params(gt_steps[i1 + k])
                    pred_action, pred_params = extract_action_and_params(
                        pred_steps[j1 + k]
                    )
                    if gt_action == pred_action:
                        if gt_params != pred_params:
                            errors.append(
                                (
                                    i1 + k + 1,
                                    f"参数错误: {gt_steps[i1 + k]} ≠ {pred_steps[j1 + k]}",
                                )
                            )
                    else:
                        errors.append(
                            (
                                i1 + k + 1,
                                f"动作类型错误: {gt_steps[i1 + k]} ≠ {pred_steps[j1 + k]}",
                            )
                        )

            elif tag == "replace":
                for k in range(i2 - i1):
                    errors.append(
                        (
                            i1 + k + 1,
                            f"步骤被错误替换: {gt_steps[i1 + k]} ≠ {pred_steps[j1 + k] if j1+k < j2 else '缺失'}",
                        )
                    )
            elif tag == "delete":
                for k in range(i1, i2):
                    errors.append((k + 1, f"缺少步骤: {gt_steps[k]}"))
            elif tag == "insert":
                # 多余步骤对应gt的最后一步之后
                # 如果插入点不在最后，则归为"上一正确步骤的下一步错误"
                nearest = max(1, i1)
                errors.append((nearest + 1, f"多余步骤: {pred_steps[j1:j2]}"))

        return errors

    # 获取错误详情并提取步骤编号
    errors = compare_plans(gt_text, pred_text)
    error_step_numbers = [error[0] for error in errors]
    return error_step_numbers


# 示例使用
if __name__ == "__main__":
    # 示例数据
    gt = """open(炸桶)
pick(食物)
move(食物, 木桌, 炸桶)
place(食物, 炸桶)
close(炸桶)
rotate(定时旋钮, 50min, 顺时针旋转225°)
rotate(温控旋钮, 200°C, 顺时针旋转270°)
wait(50min)
open(炸桶)
pick(食物)
move(食物, 炸桶, 木桌)
place(食物, 木桌)"""

    pred = """open(炸桶)
press(炸桶)
pick(食物)
move(食物, 木桌, 炸桶)
place(食物, 炸桶)
place(食物, 炸桶)
close(炸桶)
close(炸桶)
rotate(定时旋钮, 50min, 顺时针旋转226°)
rotate(温控旋钮, 200°C, 顺时针旋转270°)
open(炸桶)
pick(食物)

place(食物, 木桌)
place(食物, 木桌)"""

    # 使用主函数获取错误步骤编号
    error_numbers = find_error_steps(gt, pred)
    print("错误步骤编号：", error_numbers)
