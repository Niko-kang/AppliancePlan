#!/usr/bin/env bash
##############################################
# Train → eval loop for Data Flywheel (Qwen2.5-VL).
# Multi-round supported via MAX_ITERATIONS (default 1).
# Between-round auto data expansion is OFF by default (see ENABLE_DATA_ENHANCEMENT).
##############################################


set -euo pipefail

# Repo root (Data_Flywheel/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/eval"
ENHANCE_DIR="${REPO_ROOT}/optional/data_enhancement"
TOOLS_DIR="${REPO_ROOT}/tools"
CONFIG_DIR="${REPO_ROOT}/configs"

# ========================
# User-configurable paths (override via env or edit here)
# ========================
ROOT_PATH="${ROOT_PATH:-${REPO_ROOT}/..}"          # project workspace (contains images, etc.)
MODEL_PATH="${MODEL_PATH:-}"                        # base model or checkpoint
IMAGE_PATH="${IMAGE_PATH:-${ROOT_PATH}}"
WORK_ROOT="${WORK_ROOT:-${ROOT_PATH}/Flywheel_Train}"  # experiment outputs
TRAIN_ENTRY="${TRAIN_ENTRY:-${REPO_ROOT}/src/train/train_sft.py}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${CONFIG_DIR}/zero3.json}"
MERGE_SCRIPT="${MERGE_SCRIPT:-${TOOLS_DIR}/merge_train_data.py}"

# Multi-round: usually 1 for open release. Raise only if you intentionally iterate.
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"

# Between-round sampling from data_enhancement_* pools into next Train_data.
# Kept for research reproducibility; not needed for the standard single-round release.
ENABLE_DATA_ENHANCEMENT="${ENABLE_DATA_ENHANCEMENT:-0}"

# Multi-node cluster bootstrap is NOT shipped (sshd hardening is unsafe to publish).
# Set PET_NODE_RANK / PET_NNODES / DeepSpeed hostfile yourself when using multi-node.

CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-latest}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"


CONFIGURATIONS=(
    # "exp-name"   # each name maps to ${WORK_ROOT}/<name>
    "demo-exp"
)


# 真实测试集配置（真正的测试数据，用于最终评估）
# 真实测试数据存放在 ${BASE_DIR}/real_test_data 中（在第0轮迭代时初始化）

##################################################

# ========================
# 多机测评函数 - 主节点控制同步流程
# ========================
# 参数：
#   $1: 测试数据路径
#   $2: 结果目录
#   $3: 测试脚本名称 (test.py, test_bbox.py, test_keypage.py, test_close_loop.py)
#   $4: 测评名称 (用于日志)
multi_node_evaluate() {
    local TEST_DATA_PATH="$1"
    local RESULT_DIR="$2"
    local TEST_SCRIPT="$3"
    local EVAL_NAME="$4"
    
    # 针对BBox测评：预先抽样5%数据并拆分成24份
    # if [ "$TEST_SCRIPT" = "test_bbox.py" ]; then
    #     local ORIGINAL_TEST_DATA_PATH="$TEST_DATA_PATH"
    #     local SAMPLED_TEST_DIR="${ORIGINAL_TEST_DATA_PATH%/}_sampled"
    #     log_message "BBox测评：准备抽样数据到 ${SAMPLED_TEST_DIR}"
    #     prepare_bbox_sample_data "$ORIGINAL_TEST_DATA_PATH" "$SAMPLED_TEST_DIR"
    #     TEST_DATA_PATH="$SAMPLED_TEST_DIR"
    # fi

    # 创建结果目录（即使数据不存在，也需要同步机制）
    mkdir -p "${RESULT_DIR}"
    
    if [ ! -d "$TEST_DATA_PATH" ] || [ -z "$(ls -A "$TEST_DATA_PATH" 2>/dev/null)" ]; then
        log_message "警告: ${EVAL_NAME}测试数据目录不存在或为空: $TEST_DATA_PATH，跳过（但仍需同步所有节点）"
        
        # 即使数据不存在，也要进行同步，确保所有节点都知道跳过这个评估
        # 首先等待所有节点进入这个测评任务（确保它们已完成上一个测评）
        log_message "主节点/Worker节点：等待所有节点进入${EVAL_NAME}测评（确保上一个测评已完成）..."
        mkdir -p "${RESULT_DIR}"
        
        # 创建一个同步标记，表示当前节点已进入这个测评任务
        touch "${RESULT_DIR}/node_${PET_NODE_RANK}_entered.done"
        sync
        log_message "✓ 节点 ${PET_NODE_RANK}: 已进入${EVAL_NAME}测评任务"
        
        # 主节点等待所有节点都进入这个测评任务
        if [ ${PET_NODE_RANK} -eq 0 ]; then
            log_message "主节点：等待所有节点进入${EVAL_NAME}测评任务..."
            enter_wait_count=0
            while true; do
                sync
                entered_nodes=0
                for ((node=0; node<${PET_NNODES}; node++)); do
                    if [ -f "${RESULT_DIR}/node_${node}_entered.done" ]; then
                        entered_nodes=$((entered_nodes + 1))
                    fi
                done
                if [ $entered_nodes -ge ${PET_NNODES} ]; then
                    log_message "✓ 主节点：所有 ${PET_NNODES} 个节点都已进入${EVAL_NAME}测评任务"
                    break
                else
                    enter_wait_count=$((enter_wait_count + 1))
                    if [ $((enter_wait_count % 10)) -eq 0 ]; then
                        log_message "主节点：等待中 ($entered_nodes/${PET_NNODES} 个节点已进入)..."
                    fi
                    sleep 2
                fi
            done
            
            # 清理进入标记
            rm -f "${RESULT_DIR}"/node_*_entered.done
            sync
        else
            # Worker节点等待主节点确认所有节点都已进入（等待主节点清理所有进入标记）
            log_message "节点 ${PET_NODE_RANK}: 等待主节点确认所有节点已进入${EVAL_NAME}测评..."
            wait_count=0
            while true; do
                sync
                # 检查是否还有任何进入标记存在（如果有，说明主节点还未清理）
                entered_count=0
                for ((node=0; node<${PET_NNODES}; node++)); do
                    if [ -f "${RESULT_DIR}/node_${node}_entered.done" ]; then
                        entered_count=$((entered_count + 1))
                    fi
                done
                # 如果所有进入标记都被清理了，说明主节点已经确认所有节点都已进入
                if [ $entered_count -eq 0 ]; then
                    log_message "✓ 节点 ${PET_NODE_RANK}: 主节点已确认所有节点进入${EVAL_NAME}测评任务"
                    break
                fi
                wait_count=$((wait_count + 1))
                if [ $((wait_count % 10)) -eq 0 ]; then
                    log_message "节点 ${PET_NODE_RANK}: 仍在等待主节点确认... (已等待 $((wait_count * 2)) 秒, 还有 $entered_count 个节点标记存在)"
                fi
                sleep 2
            done
            # 清理自己的进入标记（以防万一）
            rm -f "${RESULT_DIR}/node_${PET_NODE_RANK}_entered.done"
            sync
        fi
        
        # ==================== 主节点创建同步信号 ====================
        if [ ${PET_NODE_RANK} -eq 0 ]; then
            # 创建文件计数和开始信号，让分节点知道可以跳过
            echo "0" > "${RESULT_DIR}/file_count.txt"
            touch "${RESULT_DIR}/file_list.txt"
            sync
            echo "$(date)" > "${RESULT_DIR}/inference_start.done"
            sync
            sync
            log_message "✓ 主节点：已创建跳过信号，所有节点可以退出"
            # 主节点也标记完成
            touch "${RESULT_DIR}/node_0.done"
            sync
            log_message "✓ 主节点：已标记完成"
            # 等待所有节点完成
            log_message "主节点：等待所有节点标记完成..."
            wait_count=0
            while true; do
                sync
                finished_nodes=0
                for ((node=0; node<${PET_NNODES}; node++)); do
                    if [ -f "${RESULT_DIR}/node_${node}.done" ]; then
                        finished_nodes=$((finished_nodes + 1))
                    fi
                done
                if [ $finished_nodes -ge ${PET_NNODES} ]; then
                    log_message "✓ 主节点：所有 ${PET_NNODES} 个节点都已完成"
                    break
                else
                    wait_count=$((wait_count + 1))
                    if [ $((wait_count % 6)) -eq 0 ]; then
                        log_message "主节点：等待中 ($finished_nodes/${PET_NNODES})..."
                    fi
                    sleep 2
                fi
            done
            
            # 主节点确认所有节点状态一致，准备开始汇总（数据不存在情况）
            log_message "主节点：步骤3.5 - 确认所有节点状态一致，准备开始汇总（数据不存在情况）..."
            
            # 主节点创建自己的准备标记
            touch "${RESULT_DIR}/node_0_ready_for_aggregation.done"
            sync
            
            # 等待所有分节点确认已准备好
            log_message "主节点：等待所有分节点确认已准备好汇总..."
            wait_count=0
            while true; do
                sync
                ready_nodes=1  # 主节点自己
                for ((node=1; node<${PET_NNODES}; node++)); do
                    if [ -f "${RESULT_DIR}/node_${node}_ready_for_aggregation.done" ]; then
                        ready_nodes=$((ready_nodes + 1))
                    fi
                done
                
                if [ $ready_nodes -ge ${PET_NNODES} ]; then
                    log_message "✓ 主节点：所有 ${PET_NNODES} 个节点都已准备好，可以开始汇总"
                    break
                else
                    wait_count=$((wait_count + 1))
                    if [ $((wait_count % 6)) -eq 0 ]; then
                        log_message "主节点：等待中 (${ready_nodes}/${PET_NNODES} 个节点已准备好)..."
                    fi
                    sleep 2
                fi
            done
            
            # 清理准备确认标记
            rm -f "${RESULT_DIR}"/node_*_ready_for_aggregation.done
            sync
            log_message "✓ 主节点：所有节点状态确认完成，开始汇总"
            
            # 创建汇总完成信号，让分节点知道可以退出
            touch "${RESULT_DIR}/aggregation_done.done"
            sync
            sync
            log_message "✓ 主节点：已创建退出信号"
            
            # 等待所有分节点退出（删除他们的node_X.done）
            log_message "主节点：等待所有分节点退出..."
            exit_wait_count=0
            while true; do
                sync
                remaining_nodes=0
                for ((node=1; node<${PET_NNODES}; node++)); do
                    if [ -f "${RESULT_DIR}/node_${node}.done" ]; then
                        remaining_nodes=$((remaining_nodes + 1))
                    fi
                done
                if [ $remaining_nodes -eq 0 ]; then
                    log_message "✓ 主节点：所有分节点已退出"
                    break
                else
                    exit_wait_count=$((exit_wait_count + 1))
                    if [ $((exit_wait_count % 6)) -eq 0 ]; then
                        log_message "主节点：等待分节点退出 ($remaining_nodes 个节点仍在退出中)..."
                    fi
                    sleep 2
                fi
            done
            
            # 清理所有标记文件（确保所有节点都已退出后再清理）
            log_message "主节点：清理临时文件和标记文件..."
            rm -f "${RESULT_DIR}"/file_count.txt "${RESULT_DIR}"/file_list.txt
            rm -f "${RESULT_DIR}"/inference_start.done
            rm -f "${RESULT_DIR}"/aggregation_done.done
            rm -f "${RESULT_DIR}"/node_*.done
            sync
            sync
            log_message "✓ 主节点：清理完成，退出${EVAL_NAME}测评，所有节点可以进入下一个流程"
        else
            # ==================== Worker节点等待同步 ====================
            # 等待开始信号
            log_message "节点 ${PET_NODE_RANK}: 等待主节点开始信号（数据不存在情况）..."
            wait_count=0
            while true; do
                sync
                ls "${RESULT_DIR}" > /dev/null 2>&1
                if [ -f "${RESULT_DIR}/inference_start.done" ]; then
                    break
                fi
                sleep 2
                wait_count=$((wait_count + 1))
                if [ $((wait_count % 10)) -eq 0 ]; then
                    log_message "节点 ${PET_NODE_RANK}: 仍在等待开始信号... (已等待 $((wait_count * 2)) 秒)"
                fi
            done
            log_message "✓ 节点 ${PET_NODE_RANK}: 收到开始信号"
            
            # 标记完成（NUM_FILES=0的情况）
            touch "${RESULT_DIR}/node_${PET_NODE_RANK}.done"
            sync
            log_message "✓ 节点 ${PET_NODE_RANK}: 已标记完成"
            
            # 等待所有节点完成（数据不存在情况）
            log_message "节点 ${PET_NODE_RANK}: 步骤3 - 等待所有节点完成（数据不存在情况）..."
            wait_count=0
            while true; do
                sync
                finished_nodes=0
                for ((node=0; node<${PET_NNODES}; node++)); do
                    if [ -f "${RESULT_DIR}/node_${node}.done" ]; then
                        finished_nodes=$((finished_nodes + 1))
                    fi
                done
                
                if [ $finished_nodes -ge ${PET_NNODES} ]; then
                    log_message "✓ 节点 ${PET_NODE_RANK}: 所有 ${PET_NNODES} 个节点都已完成"
                    break
                else
                    wait_count=$((wait_count + 1))
                    if [ $((wait_count % 6)) -eq 0 ]; then
                        log_message "节点 ${PET_NODE_RANK}: 等待中 ($finished_nodes/${PET_NNODES})..."
                    fi
                    sleep 5
                fi
            done
            
            # Worker节点：确认已准备好，等待主节点确认（数据不存在情况）
            log_message "节点 ${PET_NODE_RANK}: 已准备好汇总，通知主节点..."
            touch "${RESULT_DIR}/node_${PET_NODE_RANK}_ready_for_aggregation.done"
            sync
            
            # 等待主节点确认所有节点都准备好（通过检查准备标记是否被清理）
            log_message "节点 ${PET_NODE_RANK}: 等待主节点确认所有节点都准备好..."
            wait_count=0
            while true; do
                sync
                # 如果自己的准备标记被清理了，说明主节点已确认所有节点都准备好
                if [ ! -f "${RESULT_DIR}/node_${PET_NODE_RANK}_ready_for_aggregation.done" ]; then
                    log_message "✓ 节点 ${PET_NODE_RANK}: 主节点已确认所有节点都准备好"
                    break
                fi
                wait_count=$((wait_count + 1))
                if [ $((wait_count % 10)) -eq 0 ]; then
                    log_message "节点 ${PET_NODE_RANK}: 仍在等待主节点确认... (已等待 $((wait_count * 2)) 秒)"
                fi
                sleep 2
            done
            
            # 等待主节点完成汇总（重要：必须等待主节点汇总完成并发出退出信号）
            log_message "节点 ${PET_NODE_RANK}: 等待主节点完成汇总并发出退出信号..."
            wait_count=0
            while [ ! -f "${RESULT_DIR}/aggregation_done.done" ]; do
                sleep 2
                wait_count=$((wait_count + 1))
                if [ $((wait_count % 10)) -eq 0 ]; then
                    log_message "节点 ${PET_NODE_RANK}: 仍在等待主节点汇总完成... (已等待 $((wait_count * 2)) 秒)"
                fi
            done
            log_message "✓ 节点 ${PET_NODE_RANK}: 收到退出信号，准备退出${EVAL_NAME}测评"
            
            # 删除自己的完成标记，通知主节点已退出
            rm -f "${RESULT_DIR}/node_${PET_NODE_RANK}.done"
            sync
            log_message "✓ 节点 ${PET_NODE_RANK}: 已删除完成标记，退出${EVAL_NAME}测评"
            
            # 等待主节点确认所有节点都已退出（确保主节点清理完成后再进入下一个流程）
            log_message "节点 ${PET_NODE_RANK}: 等待主节点确认所有节点退出（确保流程完全结束）..."
            while [ -f "${RESULT_DIR}/aggregation_done.done" ] || [ -f "${RESULT_DIR}/inference_start.done" ]; do
                sleep 1
            done
            log_message "✓ 节点 ${PET_NODE_RANK}: 主节点已清理完成，可以进入下一个流程"
        fi
        
        return 0
    fi
    
    log_message "----------------------------------------"
    log_message "开始${EVAL_NAME}多机并行测评，节点: ${PET_NODE_RANK}/${PET_NNODES}"
    log_message "----------------------------------------"
    
    # 创建每个节点的独立结果目录
    mkdir -p "${RESULT_DIR}/node_${PET_NODE_RANK}"
    local NODE_RESULT_DIR="${RESULT_DIR}/node_${PET_NODE_RANK}"
    
    # ==================== 步骤1：主节点执行文件分发逻辑 ====================
    if [ ${PET_NODE_RANK} -eq 0 ]; then
        log_message "主节点：步骤1 - 执行文件分发逻辑..."
        
        # 获取测试文件列表（处理文件夹为空的情况）
        # 使用find命令更安全地获取文件列表，避免通配符匹配失败
        local TEST_FILES=()
        if [ -d "$TEST_DATA_PATH" ]; then
            while IFS= read -r -d '' file; do
                TEST_FILES+=("$file")
            done < <(find "$TEST_DATA_PATH" -maxdepth 1 -type f -name "*.json" -print0 2>/dev/null | sort -z)
        fi
        local NUM_FILES=${#TEST_FILES[@]}
        
        if [ $NUM_FILES -eq 0 ]; then
            log_message "警告: 未找到${EVAL_NAME}测试文件，所有节点跳过"
            # 创建文件计数和开始信号，让分节点知道可以跳过
            mkdir -p "${RESULT_DIR}"
            echo "0" > "${RESULT_DIR}/file_count.txt"
            touch "${RESULT_DIR}/file_list.txt"
            sync
            echo "$(date)" > "${RESULT_DIR}/inference_start.done"
            sync
            sync
            log_message "✓ 主节点：已创建跳过信号，所有节点可以退出"
            # 主节点也标记完成，避免分节点等待
            touch "${RESULT_DIR}/node_0.done"
            sync
            log_message "✓ 主节点：已标记完成"
            # 等待所有节点完成
            log_message "主节点：等待所有节点标记完成..."
            local wait_count=0
            while true; do
                sync
                local finished_nodes=0
                for ((node=0; node<${PET_NNODES}; node++)); do
                    if [ -f "${RESULT_DIR}/node_${node}.done" ]; then
                        finished_nodes=$((finished_nodes + 1))
                    fi
                done
                if [ $finished_nodes -ge ${PET_NNODES} ]; then
                    log_message "✓ 主节点：所有 ${PET_NNODES} 个节点都已完成"
                    break
                else
                    wait_count=$((wait_count + 1))
                    if [ $((wait_count % 6)) -eq 0 ]; then
                        log_message "主节点：等待中 ($finished_nodes/${PET_NNODES})..."
                    fi
                    sleep 2
                fi
            done
            
            # 主节点确认所有节点状态一致，准备开始汇总（文件数为0情况）
            log_message "主节点：步骤3.5 - 确认所有节点状态一致，准备开始汇总（文件数为0情况）..."
            
            # 主节点创建自己的准备标记
            touch "${RESULT_DIR}/node_0_ready_for_aggregation.done"
            sync
            
            # 等待所有分节点确认已准备好
            log_message "主节点：等待所有分节点确认已准备好汇总..."
            wait_count=0
            while true; do
                sync
                ready_nodes=1  # 主节点自己
                for ((node=1; node<${PET_NNODES}; node++)); do
                    if [ -f "${RESULT_DIR}/node_${node}_ready_for_aggregation.done" ]; then
                        ready_nodes=$((ready_nodes + 1))
                    fi
                done
                
                if [ $ready_nodes -ge ${PET_NNODES} ]; then
                    log_message "✓ 主节点：所有 ${PET_NNODES} 个节点都已准备好，可以开始汇总"
                    break
                else
                    wait_count=$((wait_count + 1))
                    if [ $((wait_count % 6)) -eq 0 ]; then
                        log_message "主节点：等待中 (${ready_nodes}/${PET_NNODES} 个节点已准备好)..."
                    fi
                    sleep 2
                fi
            done
            
            # 清理准备确认标记
            rm -f "${RESULT_DIR}"/node_*_ready_for_aggregation.done
            sync
            log_message "✓ 主节点：所有节点状态确认完成，开始汇总"
            
            # 创建汇总完成信号，让分节点知道可以退出
            touch "${RESULT_DIR}/aggregation_done.done"
            sync
            sync
            log_message "✓ 主节点：已创建退出信号"
            
            # 等待所有分节点退出（删除他们的node_X.done）
            log_message "主节点：等待所有分节点退出..."
            local exit_wait_count=0
            while true; do
                sync
                local remaining_nodes=0
                for ((node=1; node<${PET_NNODES}; node++)); do
                    if [ -f "${RESULT_DIR}/node_${node}.done" ]; then
                        remaining_nodes=$((remaining_nodes + 1))
                    fi
                done
                if [ $remaining_nodes -eq 0 ]; then
                    log_message "✓ 主节点：所有分节点已退出"
                    break
                else
                    exit_wait_count=$((exit_wait_count + 1))
                    if [ $((exit_wait_count % 6)) -eq 0 ]; then
                        log_message "主节点：等待分节点退出 ($remaining_nodes 个节点仍在退出中)..."
                    fi
                    sleep 2
                fi
            done
            
            # 清理所有标记文件（确保所有节点都已退出后再清理）
            log_message "主节点：清理临时文件和标记文件..."
            rm -f "${RESULT_DIR}"/file_count.txt "${RESULT_DIR}"/file_list.txt
            rm -f "${RESULT_DIR}"/inference_start.done
            rm -f "${RESULT_DIR}"/aggregation_done.done
            rm -f "${RESULT_DIR}"/node_*.done
            sync
            sync
            log_message "✓ 主节点：清理完成，退出${EVAL_NAME}测评，所有节点可以进入下一个流程"
            return 0
        fi
        
        log_message "主节点：找到 ${EVAL_NAME} $NUM_FILES 个测试文件"
        
        # 将文件列表保存到文件，供其他节点读取
        echo "$NUM_FILES" > "${RESULT_DIR}/file_count.txt"
        printf '%s\n' "${TEST_FILES[@]}" > "${RESULT_DIR}/file_list.txt"
        sync
        
        log_message "✓ 主节点：文件分发完成"
        
        # ==================== 步骤2：主节点给出同步开始信号 ====================
        log_message "主节点：步骤2 - 发出同步开始信号..."
        # 确保目录存在
        mkdir -p "${RESULT_DIR}"
        sync
        # 创建信号文件，写入内容确保文件真正创建
        echo "$(date)" > "${RESULT_DIR}/inference_start.done"
        # 强制同步文件系统
        sync
        # 使用 ls 强制刷新文件系统缓存
        ls -lh "${RESULT_DIR}/inference_start.done" > /dev/null 2>&1
        sync
        # 等待一小段时间，确保文件系统同步到所有节点
        sleep 1
        sync
        # 验证文件确实创建成功
        if [ -f "${RESULT_DIR}/inference_start.done" ]; then
            log_message "✓ 主节点：开始信号已发出，文件路径: ${RESULT_DIR}/inference_start.done"
            log_message "主节点：文件详细信息: $(ls -lh "${RESULT_DIR}/inference_start.done")"
            log_message "主节点：文件内容: $(cat "${RESULT_DIR}/inference_start.done")"
            log_message "主节点：文件权限: $(stat -c '%a %U:%G' "${RESULT_DIR}/inference_start.done" 2>/dev/null || echo '无法获取')"
        else
            log_message "错误：主节点无法创建开始信号文件"
            return 1
        fi
    fi
    
    # ==================== 步骤2：所有节点等待开始信号 ====================
    # 确保目录存在
    mkdir -p "${RESULT_DIR}"
    sync
    log_message "节点 ${PET_NODE_RANK}: 等待主节点开始信号，检查路径: ${RESULT_DIR}/inference_start.done"
    log_message "节点 ${PET_NODE_RANK}: RESULT_DIR 路径: ${RESULT_DIR}"
    log_message "节点 ${PET_NODE_RANK}: 目录是否存在: $([ -d "${RESULT_DIR}" ] && echo '是' || echo '否')"
    local wait_count=0
    while true; do
        # 强制同步文件系统，刷新缓存
        sync
        # 刷新目录缓存
        ls "${RESULT_DIR}" > /dev/null 2>&1
        # 检查文件是否存在
        if [ -f "${RESULT_DIR}/inference_start.done" ]; then
            break
        fi
        sleep 2
        wait_count=$((wait_count + 1))
        # 每10次循环显示详细信息
        if [ $((wait_count % 10)) -eq 0 ]; then
            log_message "节点 ${PET_NODE_RANK}: 仍在等待开始信号... (已等待 $((wait_count * 2)) 秒)"
            log_message "节点 ${PET_NODE_RANK}: 检查目录 ${RESULT_DIR} 是否存在: $([ -d "${RESULT_DIR}" ] && echo '是' || echo '否')"
            if [ -d "${RESULT_DIR}" ]; then
                log_message "节点 ${PET_NODE_RANK}: 目录内容:"
                ls -la "${RESULT_DIR}" 2>/dev/null | head -10 | while read line; do
                    log_message "  $line"
                done
            fi
            log_message "节点 ${PET_NODE_RANK}: 直接检查文件: $([ -f "${RESULT_DIR}/inference_start.done" ] && echo '存在' || echo '不存在')"
        fi
    done
    sync
    log_message "✓ 节点 ${PET_NODE_RANK}: 收到开始信号，文件路径: ${RESULT_DIR}/inference_start.done"
    log_message "节点 ${PET_NODE_RANK}: 文件内容: $(cat "${RESULT_DIR}/inference_start.done" 2>/dev/null || echo '无法读取')"
    
    # 读取文件列表和总数
    local NUM_FILES=$(cat "${RESULT_DIR}/file_count.txt" 2>/dev/null)
    if [ -z "$NUM_FILES" ] || [ "$NUM_FILES" -eq 0 ]; then
        log_message "节点 ${PET_NODE_RANK}: 没有文件需要处理，标记完成"
        touch "${RESULT_DIR}/node_${PET_NODE_RANK}.done"
        sync
        log_message "✓ 节点 ${PET_NODE_RANK}: 已标记完成"
        
        # 等待所有节点完成（文件数为0情况）
        log_message "节点 ${PET_NODE_RANK}: 步骤3 - 等待所有节点完成（文件数为0情况）..."
        wait_count=0
        while true; do
            sync
            finished_nodes=0
            for ((node=0; node<${PET_NNODES}; node++)); do
                if [ -f "${RESULT_DIR}/node_${node}.done" ]; then
                    finished_nodes=$((finished_nodes + 1))
                fi
            done
            
            if [ $finished_nodes -ge ${PET_NNODES} ]; then
                log_message "✓ 节点 ${PET_NODE_RANK}: 所有 ${PET_NNODES} 个节点都已完成"
                break
            else
                wait_count=$((wait_count + 1))
                if [ $((wait_count % 6)) -eq 0 ]; then
                    log_message "节点 ${PET_NODE_RANK}: 等待中 ($finished_nodes/${PET_NNODES})..."
                fi
                sleep 5
            fi
        done
        
        # Worker节点：确认已准备好，等待主节点确认（文件数为0情况）
        log_message "节点 ${PET_NODE_RANK}: 已准备好汇总，通知主节点..."
        touch "${RESULT_DIR}/node_${PET_NODE_RANK}_ready_for_aggregation.done"
        sync
        
        # 等待主节点确认所有节点都准备好（通过检查准备标记是否被清理）
        log_message "节点 ${PET_NODE_RANK}: 等待主节点确认所有节点都准备好..."
        wait_count=0
        while true; do
            sync
            # 如果自己的准备标记被清理了，说明主节点已确认所有节点都准备好
            if [ ! -f "${RESULT_DIR}/node_${PET_NODE_RANK}_ready_for_aggregation.done" ]; then
                log_message "✓ 节点 ${PET_NODE_RANK}: 主节点已确认所有节点都准备好"
                break
            fi
            wait_count=$((wait_count + 1))
            if [ $((wait_count % 10)) -eq 0 ]; then
                log_message "节点 ${PET_NODE_RANK}: 仍在等待主节点确认... (已等待 $((wait_count * 2)) 秒)"
            fi
            sleep 2
        done
        
        # 等待主节点完成汇总并发出退出信号
        log_message "节点 ${PET_NODE_RANK}: 等待主节点完成汇总并发出退出信号..."
        wait_count=0
        while [ ! -f "${RESULT_DIR}/aggregation_done.done" ]; do
            sleep 2
            wait_count=$((wait_count + 1))
            if [ $((wait_count % 10)) -eq 0 ]; then
                log_message "节点 ${PET_NODE_RANK}: 仍在等待主节点汇总完成... (已等待 $((wait_count * 2)) 秒)"
            fi
        done
        log_message "✓ 节点 ${PET_NODE_RANK}: 收到退出信号，准备退出${EVAL_NAME}测评"
        
        # 删除自己的完成标记，通知主节点已退出
        rm -f "${RESULT_DIR}/node_${PET_NODE_RANK}.done"
        sync
        log_message "✓ 节点 ${PET_NODE_RANK}: 已删除完成标记，退出${EVAL_NAME}测评"
        
        # 等待主节点确认所有节点都已退出（确保主节点清理完成后再进入下一个流程）
        log_message "节点 ${PET_NODE_RANK}: 等待主节点确认所有节点退出（确保流程完全结束）..."
        while [ -f "${RESULT_DIR}/aggregation_done.done" ] || [ -f "${RESULT_DIR}/inference_start.done" ]; do
            sleep 1
        done
        log_message "✓ 节点 ${PET_NODE_RANK}: 主节点已清理完成，可以进入下一个流程"
        
        return 0
    else
        local TEST_FILES=($(cat "${RESULT_DIR}/file_list.txt" 2>/dev/null))
        
        # ==================== 步骤2：所有节点开始执行 ====================
        # 为当前节点分配文件 (轮询分配)
        local NODE_FILES=()
        for ((i=0; i<$NUM_FILES; i++)); do
            if [ $((i % ${PET_NNODES})) -eq ${PET_NODE_RANK} ]; then
                NODE_FILES+=("${TEST_FILES[$i]}")
            fi
        done
        
        log_message "节点 ${PET_NODE_RANK}: 分配了 ${#NODE_FILES[@]} 个${EVAL_NAME}文件"
        
        # 执行当前节点的推理
        if [ ${#NODE_FILES[@]} -gt 0 ]; then
            # 创建临时目录用于当前节点的测试文件
            local NODE_TEST_DIR="${TEST_DATA_PATH}_node_${PET_NODE_RANK}_temp"
            mkdir -p "$NODE_TEST_DIR"
            
            # 复制分配的文件到临时目录
            for file in "${NODE_FILES[@]}"; do
                if [ -f "$file" ]; then
                    cp "$file" "$NODE_TEST_DIR/"
                fi
            done
            
            # 获取模型路径
            if [ ${PET_NODE_RANK} -ne 0 ]; then
                if [ -f "${ITERATION_DIR}/log/current_checkpoint.txt" ]; then
                    MODEL_CHECKPOINT=$(cat "${ITERATION_DIR}/log/current_checkpoint.txt")
                else
                    MODEL_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/checkpoint-*/ 2>/dev/null | head -n 1)
                fi
            fi
            
            # 设置错误处理，确保即使推理崩溃也会创建done文件
            create_done_file_on_exit() {
                local exit_code=$?
                log_message "节点 ${PET_NODE_RANK}: 推理进程异常退出，退出码: ${exit_code}"
                # 无论成功还是失败，都创建done文件，防止其他节点无限等待
                if [ ! -f "${RESULT_DIR}/node_${PET_NODE_RANK}.done" ]; then
                    log_message "节点 ${PET_NODE_RANK}: 推理异常退出，强制创建完成标记文件以防止死锁..."
                    touch "${RESULT_DIR}/node_${PET_NODE_RANK}.done"
                    sync
                    sync
                    log_message "✓ 节点 ${PET_NODE_RANK}: 已创建完成标记文件（异常退出）"
                fi
                # 清理临时目录（以防万一）
                [ -d "$NODE_TEST_DIR" ] && rm -rf "$NODE_TEST_DIR" 2>/dev/null || true
            }
            trap create_done_file_on_exit EXIT INT TERM ERR
            
            # 执行推理
            log_message "节点 ${PET_NODE_RANK}: 开始${EVAL_NAME}推理..."
            local inference_exit_code=0
            python "${EVAL_DIR}/${TEST_SCRIPT}" \
                --model_name "${MODEL_CHECKPOINT}" \
                --data_path "$NODE_TEST_DIR" \
                --root_path "${ROOT_PATH}" \
                --result_dir "${NODE_RESULT_DIR}" \
                2>&1 | tee -a "$MAIN_LOG_FILE"
            inference_exit_code=${PIPESTATUS[0]}
            
            # 清理临时目录
            rm -rf "$NODE_TEST_DIR"
            
            if [ $inference_exit_code -eq 0 ]; then
                log_message "✓ 节点 ${PET_NODE_RANK}: ${EVAL_NAME}推理完成（退出码: 0）"
            else
                log_message "警告: 节点 ${PET_NODE_RANK}: ${EVAL_NAME}推理失败（退出码: ${inference_exit_code}），但继续执行以避免死锁"
            fi
            
            # 清理trap（推理已完成，done文件会在后面统一创建）
            trap - EXIT INT TERM ERR
        else
            log_message "节点 ${PET_NODE_RANK}: 没有分配到文件"
        fi
        
        # 标记当前节点推理完成（无论是否分配到文件都要创建，防止死锁）
        log_message "节点 ${PET_NODE_RANK}: 正在创建完成标记文件: ${RESULT_DIR}/node_${PET_NODE_RANK}.done"
        touch "${RESULT_DIR}/node_${PET_NODE_RANK}.done"
        sync
        sync
        # 验证文件是否成功创建
        if [ -f "${RESULT_DIR}/node_${PET_NODE_RANK}.done" ]; then
            log_message "✓ 节点 ${PET_NODE_RANK}: 已标记完成"
        else
            log_message "错误: 节点 ${PET_NODE_RANK}: 无法创建完成标记文件，重试..."
            sleep 1
            touch "${RESULT_DIR}/node_${PET_NODE_RANK}.done"
            sync
            sync
            if [ -f "${RESULT_DIR}/node_${PET_NODE_RANK}.done" ]; then
                log_message "✓ 节点 ${PET_NODE_RANK}: 重试后已标记完成"
            else
                log_message "严重错误: 节点 ${PET_NODE_RANK}: 重试后仍无法创建完成标记文件"
            fi
        fi
    fi
    
    # ==================== 步骤3：所有完成节点等待 ====================
    log_message "节点 ${PET_NODE_RANK}: 步骤3 - 等待所有节点完成..."
    local wait_count=0
    while true; do
        sync
        # 强制刷新目录缓存
        ls "${RESULT_DIR}" > /dev/null 2>&1
        local finished_nodes=0
        local node_status=""
        for ((node=0; node<${PET_NNODES}; node++)); do
            if [ -f "${RESULT_DIR}/node_${node}.done" ]; then
                finished_nodes=$((finished_nodes + 1))
                node_status="${node_status}✓${node} "
            else
                node_status="${node_status}✗${node} "
            fi
        done
        
        if [ $finished_nodes -ge ${PET_NNODES} ]; then
            log_message "✓ 节点 ${PET_NODE_RANK}: 所有 ${PET_NNODES} 个节点都已完成"
            break
        else
            wait_count=$((wait_count + 1))
            if [ $((wait_count % 6)) -eq 0 ]; then
                log_message "节点 ${PET_NODE_RANK}: 等待中 ($finished_nodes/${PET_NNODES})... [节点状态: ${node_status}]"
            fi
            sleep 5
        fi
    done
    
    # ==================== 步骤3.5：主节点确认所有节点状态一致，准备开始汇总 ====================
    if [ ${PET_NODE_RANK} -eq 0 ]; then
        log_message "主节点：步骤3.5 - 确认所有节点状态一致，准备开始汇总..."
        
        # 主节点创建自己的准备标记
        touch "${RESULT_DIR}/node_0_ready_for_aggregation.done"
        sync
        
        # 等待所有分节点确认已准备好
        log_message "主节点：等待所有分节点确认已准备好汇总..."
        wait_count=0
        while true; do
            sync
            ready_nodes=1  # 主节点自己
            for ((node=1; node<${PET_NNODES}; node++)); do
                if [ -f "${RESULT_DIR}/node_${node}_ready_for_aggregation.done" ]; then
                    ready_nodes=$((ready_nodes + 1))
                fi
            done
            
            if [ $ready_nodes -ge ${PET_NNODES} ]; then
                log_message "✓ 主节点：所有 ${PET_NNODES} 个节点都已准备好，可以开始汇总"
                break
            else
                wait_count=$((wait_count + 1))
                if [ $((wait_count % 6)) -eq 0 ]; then
                    log_message "主节点：等待中 (${ready_nodes}/${PET_NNODES} 个节点已准备好)..."
                fi
                sleep 2
            fi
        done
        
        # 清理准备确认标记
        rm -f "${RESULT_DIR}"/node_*_ready_for_aggregation.done
        sync
        log_message "✓ 主节点：所有节点状态确认完成，开始汇总"
    else
        # Worker节点：确认已准备好，等待主节点确认
        log_message "节点 ${PET_NODE_RANK}: 已准备好汇总，通知主节点..."
        touch "${RESULT_DIR}/node_${PET_NODE_RANK}_ready_for_aggregation.done"
        sync
        
        # 等待主节点确认所有节点都准备好（通过检查准备标记是否被清理）
        log_message "节点 ${PET_NODE_RANK}: 等待主节点确认所有节点都准备好..."
        wait_count=0
        while true; do
            sync
            # 如果自己的准备标记被清理了，说明主节点已确认所有节点都准备好
            if [ ! -f "${RESULT_DIR}/node_${PET_NODE_RANK}_ready_for_aggregation.done" ]; then
                log_message "✓ 节点 ${PET_NODE_RANK}: 主节点已确认所有节点都准备好"
                break
            fi
            wait_count=$((wait_count + 1))
            if [ $((wait_count % 10)) -eq 0 ]; then
                log_message "节点 ${PET_NODE_RANK}: 仍在等待主节点确认... (已等待 $((wait_count * 2)) 秒)"
            fi
            sleep 2
        done
    fi
    
    # ==================== 步骤4：主节点执行汇总，其他节点退出 ====================
    if [ ${PET_NODE_RANK} -eq 0 ]; then
        log_message "----------------------------------------"
        log_message "主节点：步骤4 - 开始汇总所有节点${EVAL_NAME}结果..."
        log_message "----------------------------------------"
        
        # 合并所有节点的结果
        log_message "主节点：开始合并所有节点的结果文件..."
        for node_dir in "${RESULT_DIR}"/node_*/; do
            if [ -d "$node_dir" ]; then
                node_num=$(basename "$node_dir" | sed 's/node_//')
                log_message "合并节点 ${node_num} 的结果..."
                file_count=$(find "$node_dir" -type f 2>/dev/null | wc -l)
                log_message "节点 ${node_num}: 发现 ${file_count} 个文件需要合并"
                if [ $file_count -gt 0 ]; then
                    # 列出所有文件并复制（递归查找，支持子目录）
                    local copied_count=0
                    local failed_count=0
                    while IFS= read -r -d '' file; do
                        if [ -f "$file" ]; then
                            filename=$(basename "$file")
                            # 检查目标文件是否已存在（避免覆盖）
                            if [ -f "${RESULT_DIR}/${filename}" ]; then
                                log_message "警告: 文件已存在，跳过: ${filename} (节点 ${node_num})"
                            else
                                if cp -v "$file" "${RESULT_DIR}/" > /dev/null 2>&1; then
                                    copied_count=$((copied_count + 1))
                                    if [ $copied_count -le 3 ]; then
                                        log_message "  → 已复制: ${filename}"
                                    fi
                                else
                                    failed_count=$((failed_count + 1))
                                    log_message "错误: 复制文件失败: ${file}"
                                fi
                            fi
                        fi
                    done < <(find "$node_dir" -type f -print0 2>/dev/null)
                    log_message "✓ 节点 ${node_num}: 成功复制 ${copied_count} 个文件"
                    if [ $failed_count -gt 0 ]; then
                        log_message "警告: 节点 ${node_num}: ${failed_count} 个文件复制失败"
                    fi
                else
                    log_message "警告: 节点 ${node_num} 目录为空"
                fi
            fi
        done
        
        # 验证合并后的文件
        log_message "主节点：验证合并结果..."
        local total_source_files=0
        for node_dir in "${RESULT_DIR}"/node_*/; do
            if [ -d "$node_dir" ]; then
                local node_file_count=$(find "$node_dir" -type f 2>/dev/null | wc -l)
                total_source_files=$((total_source_files + node_file_count))
            fi
        done
        local total_merged_files=$(find "${RESULT_DIR}" -maxdepth 1 -type f \( -name "*.jsonl" -o -name "*.json" -o -name "*.txt" -o -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
        log_message "✓ 合并完成统计:"
        log_message "  - 源文件总数: ${total_source_files} 个（所有节点）"
        log_message "  - 合并后文件数: ${total_merged_files} 个"
        if [ $total_merged_files -lt $total_source_files ]; then
            log_message "警告: 合并后文件数少于源文件数，可能有文件未被正确复制或存在文件名冲突"
        fi
        log_message "文件列表（前10个）: $(find "${RESULT_DIR}" -maxdepth 1 -type f \( -name "*_inference_*.jsonl" -o -name "*_detailed_*.jsonl" -o -name "*.json" -o -name "*.png" \) -exec basename {} \; 2>/dev/null | head -10 | tr '\n' ' ')"
        
        log_message "✓ ${EVAL_NAME}多机测评完成，开始汇总统计..."
        
        # 调用汇总统计脚本前，验证是否有结果文件
        log_message "主节点：验证汇总前的结果文件..."
        local jsonl_count=$(find "${RESULT_DIR}" -maxdepth 1 -type f -name "*_inference_*.jsonl" 2>/dev/null | wc -l)
        log_message "主节点：找到 ${jsonl_count} 个推理结果文件（*_inference_*.jsonl）"
        
        if [ $jsonl_count -eq 0 ]; then
            log_message "警告: 未找到推理结果文件，跳过汇总统计"
        else
            # 显示文件列表
            log_message "结果文件列表: $(find "${RESULT_DIR}" -maxdepth 1 -type f -name "*_inference_*.jsonl" -exec basename {} \; 2>/dev/null | head -5 | tr '\n' ' ')"
            
            # 调用汇总统计脚本
            log_message "----------------------------------------"
            log_message "开始生成${EVAL_NAME}汇总统计报告..."
            log_message "汇总脚本: ${EVAL_DIR}/aggregate_eval_results.py"
            log_message "参数: --test_script ${TEST_SCRIPT} --result_dir ${RESULT_DIR}"
            log_message "----------------------------------------"
            
            local aggregate_exit_code=0
            python3 "${EVAL_DIR}/aggregate_eval_results.py" \
                --test_script "${TEST_SCRIPT}" \
                --result_dir "${RESULT_DIR}" \
                --test_data_path "${TEST_DATA_PATH}" \
                --model_name "${MODEL_CHECKPOINT}" \
                --root_path "${ROOT_PATH}" \
                --eval_name "${EVAL_NAME}" \
                2>&1 | tee -a "$MAIN_LOG_FILE"
            aggregate_exit_code=${PIPESTATUS[0]}
            
            if [ $aggregate_exit_code -eq 0 ]; then
                log_message "✓ ${EVAL_NAME}汇总统计完成（退出码: ${aggregate_exit_code}）"
                
                # 验证汇总统计生成的文件
                local summary_files=$(find "${RESULT_DIR}" -maxdepth 1 -type f \( -name "*summary*.txt" -o -name "*report*.txt" -o -name "*detailed*.jsonl" \) 2>/dev/null | wc -l)
                if [ $summary_files -gt 0 ]; then
                    log_message "✓ 汇总统计文件已生成: ${summary_files} 个"
                    log_message "汇总文件列表: $(find "${RESULT_DIR}" -maxdepth 1 -type f \( -name "*summary*.txt" -o -name "*report*.txt" -o -name "*detailed*.jsonl" \) -exec basename {} \; 2>/dev/null | tr '\n' ' ')"
                else
                    log_message "警告: 未找到汇总统计文件，可能未生成"
                fi
            else
                log_message "错误: ${EVAL_NAME}汇总统计失败（退出码: ${aggregate_exit_code}），请检查日志"
                log_message "汇总脚本路径: ${EVAL_DIR}/aggregate_eval_results.py"
                log_message "结果目录: ${RESULT_DIR}"
                log_message "检查结果目录是否存在: $([ -d "${RESULT_DIR}" ] && echo '是' || echo '否')"
            fi
        fi
        
        # 发布退出信号
        touch "${RESULT_DIR}/aggregation_done.done"
        sync
        log_message "✓ 主节点：汇总完成，已发布退出信号"
    else
        # worker节点：等待退出信号后退出（重要：必须等待主节点汇总完成并发出退出信号）
        log_message "节点 ${PET_NODE_RANK}: 等待主节点完成汇总并发出退出信号..."
        wait_count=0
        while [ ! -f "${RESULT_DIR}/aggregation_done.done" ]; do
            sleep 2
            wait_count=$((wait_count + 1))
            if [ $((wait_count % 10)) -eq 0 ]; then
                log_message "节点 ${PET_NODE_RANK}: 仍在等待主节点汇总完成... (已等待 $((wait_count * 2)) 秒)"
            fi
        done
        log_message "✓ 节点 ${PET_NODE_RANK}: 收到退出信号，准备退出${EVAL_NAME}测评"
        
        # 删除自己的完成标记，通知主节点已退出
        rm -f "${RESULT_DIR}/node_${PET_NODE_RANK}.done"
        sync
        log_message "✓ 节点 ${PET_NODE_RANK}: 已删除完成标记，退出${EVAL_NAME}测评"
        
        # 等待主节点确认所有节点都已退出（确保主节点清理完成后再进入下一个流程）
        log_message "节点 ${PET_NODE_RANK}: 等待主节点确认所有节点退出（确保流程完全结束）..."
        while [ -f "${RESULT_DIR}/aggregation_done.done" ] || [ -f "${RESULT_DIR}/inference_start.done" ]; do
            sleep 1
        done
        log_message "✓ 节点 ${PET_NODE_RANK}: 主节点已清理完成，可以进入下一个流程"
    fi
    
    # ==================== 步骤5：主节点等待所有节点退出 ====================
    if [ ${PET_NODE_RANK} -eq 0 ]; then
        log_message "主节点：步骤5 - 等待所有节点退出..."
        local exit_wait_count=0
        while true; do
            sync
            # 检查是否还有节点标记文件存在（说明节点还未完全退出）
            local remaining_nodes=0
            for ((node=1; node<${PET_NNODES}; node++)); do
                if [ -f "${RESULT_DIR}/node_${node}.done" ]; then
                    remaining_nodes=$((remaining_nodes + 1))
                fi
            done
            
            if [ $remaining_nodes -eq 0 ]; then
                log_message "✓ 主节点：所有节点已退出${EVAL_NAME}测评"
                break
            else
                exit_wait_count=$((exit_wait_count + 1))
                if [ $((exit_wait_count % 6)) -eq 0 ]; then
                    log_message "主节点：等待节点退出 ($remaining_nodes 个节点仍在退出中)..."
                fi
                sleep 3
            fi
        done
        
        # 清理文件（确保所有节点都已退出后再清理）
        log_message "主节点：清理临时文件和标记文件..."
        rm -f "${RESULT_DIR}"/file_count.txt "${RESULT_DIR}"/file_list.txt
        rm -f "${RESULT_DIR}"/inference_start.done
        rm -f "${RESULT_DIR}"/aggregation_done.done
        # 清理所有节点的done标记文件
        rm -f "${RESULT_DIR}"/node_*.done
        sync
        sync
        log_message "✓ 主节点：清理完成（包括所有节点标记文件），所有节点可以进入下一个流程"
    fi
    
    # 所有节点（包括主节点和worker节点）都确保流程完全结束后才继续
    log_message "✓ 节点 ${PET_NODE_RANK}: ${EVAL_NAME}测评流程完全结束，可以进入下一个测评任务"
}

##################################################

# ========================

##################################################
# Runtime defaults for single-node / user-managed multi-node
SHARE_DIR="${SHARE_DIR:-/tmp}"
JOYBUILDER_JOB_TEMPD="${JOYBUILDER_JOB_TEMPD:-${SHARE_DIR}/.flywheel_temp/local}"
mkdir -p "${JOYBUILDER_JOB_TEMPD}"
if [[ ! -f "${JOYBUILDER_JOB_TEMPD}/host.txt" ]]; then
    echo "$(hostname) slots=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)" > "${JOYBUILDER_JOB_TEMPD}/host.txt"
fi
export PET_NODE_RANK="${PET_NODE_RANK:-0}"
export PET_NNODES="${PET_NNODES:-1}"

# ROOT_PATH / MODEL_PATH / IMAGE_PATH already set above (env-overridable)
if [[ -z "${MODEL_PATH}" ]]; then
    echo "[ERROR] Please set MODEL_PATH to your Qwen2.5-VL base model or checkpoint" >&2
    exit 1
fi

# 训练参数
GLOBAL_BATCH_SIZE=512
BATCH_PER_DEVICE=6
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
BATCH_PER_DEVICE_EVAL=1

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/src/train:${PYTHONPATH:-}"

# ========================
# 简化版训练循环
# 包含三个独立脚本：
# 1. error_analysis.py - 错误分析脚本
# 2. data_enhancement.py - 数据增强脚本
# 3. find_error.py - 错误检测函数
# ========================

# 循环处理每个配置
for config in "${CONFIGURATIONS[@]}"; do
    # 设置主日志文件
    BASE_DIR="${WORK_ROOT}/${config}"
    mkdir -p "${BASE_DIR}"
    MAIN_LOG_FILE="${BASE_DIR}/training_pipeline_$(date +%Y%m%d_%H%M%S).log"
    
    # 日志函数
    log_message() {
        local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
        echo "$message" | tee -a "$MAIN_LOG_FILE"
    }
    
    log_message "================================================"
    log_message "开始处理配置: ${config}"
    log_message "主日志文件: ${MAIN_LOG_FILE}"
    log_message "================================================"
    
    # 初始化迭代变量
    iteration=0
    
    # 训练循环
    while [ $iteration -lt $MAX_ITERATIONS ]; do
        log_message "----------------------------------------"
        log_message "开始第 $((iteration + 1)) 轮训练 (Iteration ${iteration})"
        log_message "----------------------------------------"
        
        # 数据准备与检查仅在主节点执行；worker跳过并等待训练完成
        ITERATION_DIR="${BASE_DIR}/iteration_${iteration}"
        RESULT_DIR="${ITERATION_DIR}/result/"
        
        # 训练完成标记文件保存在主迭代目录中，所有节点可见
        TRAIN_DONE_FILE="${ITERATION_DIR}/training_done"

        if [ ${PET_NODE_RANK} == 0 ]; then
            # 创建迭代目录结构
            mkdir -p "${ITERATION_DIR}/Train_data"     # 训练数据
            mkdir -p "${ITERATION_DIR}/Test_data"      # 测试数据
            mkdir -p "${ITERATION_DIR}/plan_test"       # plan测试数据
            mkdir -p "${ITERATION_DIR}/close_loop_test"  # 闭环测试数据
            mkdir -p "${ITERATION_DIR}/model"          # 模型输出
            mkdir -p "${ITERATION_DIR}/log"            # 日志文件
            mkdir -p "${ITERATION_DIR}/result"         # 测试结果
            
            # 检查当前迭代数据是否已准备
            TRAIN_DATA_DIR="${ITERATION_DIR}/Train_data"
            TEST_DATA_DIR="${ITERATION_DIR}/Test_data"
            
            # 检查训练数据是否已准备
            if [ ! -d "${TRAIN_DATA_DIR}" ] || [ -z "$(ls -A "${TRAIN_DATA_DIR}")" ]; then
                if [ $iteration -eq 0 ]; then
                    log_message "错误: 第0轮训练数据未手动准备，请先准备数据到: $TRAIN_DATA_DIR"
                else
                    log_message "错误: 第${iteration}轮训练数据未准备，请检查上一轮的数据增强是否完成"
                fi
                exit 1
            else
                log_message "✓ 使用第${iteration}轮训练数据: $TRAIN_DATA_DIR"
            fi
            
            # 检查测试数据是否已准备
            if [ ! -d "${TEST_DATA_DIR}" ] || [ -z "$(ls -A "${TEST_DATA_DIR}")" ]; then
                if [ $iteration -eq 0 ]; then
                    log_message "错误: 第0轮测试数据未手动准备，请先准备数据到: $TEST_DATA_DIR"
                else
                    log_message "错误: 第${iteration}轮测试数据未准备，请检查上一轮的数据复制是否完成"
                fi
                exit 1
            else
                log_message "✓ 使用第${iteration}轮测试数据: $TEST_DATA_DIR"
            fi
            
            # 合并训练与测试数据
            log_message "开始合并训练数据..."
            python "${MERGE_SCRIPT}" \
                --input_dir "$TRAIN_DATA_DIR" \
                --output_file "${ITERATION_DIR}/train.json" \
                --seed 42 2>&1 | tee -a "$MAIN_LOG_FILE"

            log_message "开始合并测试数据..."
            python "${MERGE_SCRIPT}" \
                --input_dir "$TEST_DATA_DIR" \
                --output_file "${ITERATION_DIR}/test.json" \
                --seed 42 2>&1 | tee -a "$MAIN_LOG_FILE"
            
            # 训练配置
            JSON_PATH="${ITERATION_DIR}/train.json"
            VAL_JSON_PATH="${ITERATION_DIR}/test.json"
            OUTPUT_DIR="${ITERATION_DIR}/model"
            LOG_FILE="${ITERATION_DIR}/log/train.txt"
            
            # 创建输出目录
            mkdir -p "$OUTPUT_DIR"
            mkdir -p "$(dirname "$LOG_FILE")"
            
            log_message "训练输出目录: $OUTPUT_DIR"
            log_message "训练日志文件: $LOG_FILE"
        else
            log_message "Worker节点跳过数据检查与合并，等待rank0训练完成..."
        fi
        
        ####################
        cat ${JOYBUILDER_JOB_TEMPD}/host.txt
        
        # 执行训练命令
        if [ ${PET_NODE_RANK} == 0 ]; then
            log_message "=========================================="
            log_message "开始执行训练 - Iteration ${iteration}"
            log_message "=========================================="
            
            # 训练前诊断：检查系统资源
            log_message "系统资源检查:"
            log_message "  GPU 数量: $(nvidia-smi -L | wc -l)"
            log_message "  GPU 内存使用:"
            nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
                awk '{printf "    GPU %s: %s / %s MB (%.1f%%)\n", $1, $2, $3, ($2/$3)*100}' | \
                tee -a "$MAIN_LOG_FILE"
            log_message "  系统内存:"
            free -h | grep Mem | awk '{print "    总内存: " $2 ", 已用: " $3 ", 可用: " $7}' | tee -a "$MAIN_LOG_FILE"
            log_message "  训练参数:"
            log_message "    GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
            log_message "    BATCH_PER_DEVICE: $BATCH_PER_DEVICE"
            log_message "    GRAD_ACCUM_STEPS: $GRAD_ACCUM_STEPS"
            log_message "    NUM_DEVICES: $NUM_DEVICES"
            
            # 检查是否有足够的 GPU 内存
            AVAILABLE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
            if [ "$AVAILABLE_MEM" -lt 5000 ]; then
                log_message "警告: GPU 可用内存较少 ($AVAILABLE_MEM MB)，可能出现 OOM"
                log_message "建议: 考虑减少 BATCH_PER_DEVICE 或增加 GRAD_ACCUM_STEPS"
            fi
            
            # 清理上次运行的training_done标记（保存在迭代目录）
            log_message "主节点：清理上次运行的training_done标记... (${TRAIN_DONE_FILE})"
            rm -f "${TRAIN_DONE_FILE}"
            sync
            log_message "✓ 主节点：已清理training_done标记，开始训练..."
            
            deepspeed --hostfile=${JOYBUILDER_JOB_TEMPD}/host.txt \
                "${TRAIN_ENTRY}" \
                --use_liger True \
                --deepspeed "${DEEPSPEED_CONFIG}" \
                --model_id "$MODEL_PATH" \
                --data_path "$JSON_PATH" \
                --image_folder "$IMAGE_PATH" \
                --remove_unused_columns False \
                --freeze_vision_tower True \
                --freeze_llm False \
                --freeze_merger False \
                --bf16 True \
                --fp16 False \
                --disable_flash_attn2 False \
                --output_dir "$OUTPUT_DIR" \
                --num_train_epochs "$NUM_TRAIN_EPOCHS" \
                --per_device_train_batch_size "$BATCH_PER_DEVICE" \
                --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
                --image_min_pixels $((512 * 28 * 28)) \
                --image_max_pixels $((1280 * 28 * 28)) \
                --learning_rate 1e-5 \
                --merger_lr 1e-5 \
                --vision_lr 2e-6 \
                --weight_decay 0.1 \
                --warmup_ratio 0.003 \
                --lr_scheduler_type "cosine" \
                --logging_steps 1 \
                --tf32 True \
                --gradient_checkpointing True \
                --report_to tensorboard \
                --lazy_preprocess True \
                --save_strategy "epoch" \
                --save_only_model True \
                --save_total_limit 50 \
                --metric_for_best_model "eval_loss" \
                --dataloader_num_workers 8 \
                --valid_path "$VAL_JSON_PATH" \
                --eval_strategy "epoch" \
                --per_device_eval_batch_size "$BATCH_PER_DEVICE_EVAL" \
                2>&1 | tee "$LOG_FILE" | tee -a "$MAIN_LOG_FILE"
            
            log_message "✓ 第 $((iteration + 1)) 轮训练完成"
            # 标记训练完成（写入迭代目录）
            log_message "主节点：创建training_done标记文件到迭代目录... (${TRAIN_DONE_FILE})"
            # 创建标记文件，写入内容确保文件真正创建
            echo "$(date)" > "${TRAIN_DONE_FILE}"
            # 强制同步文件系统
            sync
            sync
            # 使用ls强制刷新文件系统缓存
            ls -lh "${TRAIN_DONE_FILE}" > /dev/null 2>&1
            sync
            # 验证文件确实创建成功
            if [ -f "${TRAIN_DONE_FILE}" ]; then
                log_message "✓ 主节点：训练完成标记已创建，文件路径: ${TRAIN_DONE_FILE}"
                log_message "主节点：文件详细信息: $(ls -lh "${TRAIN_DONE_FILE}" 2>/dev/null || echo '无法获取')"
                log_message "主节点：文件内容: $(cat "${TRAIN_DONE_FILE}" 2>/dev/null || echo '无法读取')"
            else
                log_message "错误：主节点无法创建训练完成标记文件"
            fi
            
            # 等待所有分节点退出等待状态
            log_message "主节点：等待所有分节点退出等待状态..."
            wait_count=0
            while true; do
                sync
                ready_nodes=0
                # 确保PET_NNODES已定义
                if [ -z "${PET_NNODES}" ]; then
                    log_message "错误: PET_NNODES未定义"
                    break
                fi
                total_workers=$((PET_NNODES - 1))
                for ((node=1; node<${PET_NNODES}; node++)); do
                    if [ -f "${ITERATION_DIR}/node_${node}_training_ready.done" ]; then
                        ready_nodes=$((ready_nodes + 1))
                    fi
                done
                
                if [ "${ready_nodes}" -ge "${total_workers}" ]; then
                    log_message "✓ 主节点：所有 ${total_workers} 个分节点都已退出等待状态"
                    break
                else
                    wait_count=$((wait_count + 1))
                    if [ $((wait_count % 6)) -eq 0 ]; then
                        log_message "主节点：等待中 (${ready_nodes}/${total_workers} 个分节点已退出等待状态)..."
                    fi
                    sleep 2
                fi
            done
            
            # 清除训练完成标志
            log_message "主节点：清除训练完成标志..."
            rm -f "${TRAIN_DONE_FILE}"
            # 清除所有分节点的ready标记
            rm -f "${ITERATION_DIR}"/node_*_training_ready.done
            sync
            log_message "✓ 主节点：训练完成标志已清除"
        else
            # worker节点等待训练结束
            log_message "Worker节点等待rank0训练完成..."
            log_message "检查training_done标记文件: ${TRAIN_DONE_FILE}"
            
            # 等待训练完成标记
            local wait_count=0
            while [ ! -f "${TRAIN_DONE_FILE}" ]; do
                sync
                # 使用ls强制刷新文件系统缓存
                ls -lh "${ITERATION_DIR}" > /dev/null 2>&1
                wait_count=$((wait_count + 1))
                if [ $((wait_count % 6)) -eq 0 ]; then
                    log_message "Worker节点 ${PET_NODE_RANK}: 仍在等待rank0训练完成... (已等待 $((wait_count * 10)) 秒)"
                    log_message "Worker节点 ${PET_NODE_RANK}: 检查标记文件路径: ${TRAIN_DONE_FILE}"
                    log_message "Worker节点 ${PET_NODE_RANK}: 检查迭代目录是否存在: $([ -d \"${ITERATION_DIR}\" ] && echo '是' || echo '否')"
                    log_message "Worker节点 ${PET_NODE_RANK}: 检查标记文件是否存在: $([ -f \"${TRAIN_DONE_FILE}\" ] && echo '是' || echo '否')"
                    if [ -d "${ITERATION_DIR}" ]; then
                        log_message "Worker节点 ${PET_NODE_RANK}: 迭代目录内容: $(ls -la "${ITERATION_DIR}" 2>/dev/null | head -10 | tr '\n' ';')"
                    fi
                fi
                sleep 10
            done
            sync
            sync
            log_message "✓ Worker节点 ${PET_NODE_RANK}: 检测到rank0训练完成标记，准备进入推理环境"
            log_message "Worker节点 ${PET_NODE_RANK}: 文件内容: $(cat "${TRAIN_DONE_FILE}" 2>/dev/null || echo '无法读取')"
            
            # Worker节点标记已退出等待状态
            worker_ready_file="${ITERATION_DIR}/node_${PET_NODE_RANK}_training_ready.done"
            touch "${worker_ready_file}"
            sync
            log_message "✓ Worker节点 ${PET_NODE_RANK}: 已标记退出等待状态"
        fi
        
        # ========================
        # 进入conda infer环境用于推理（所有节点都要执行）
        # ========================
        log_message "=========================================="
        log_message "进入conda infer环境（带vllm）用于推理"
        log_message "=========================================="
        source /opt/miniconda/etc/profile.d/conda.sh
        conda activate infer
        log_message "✓ 当前Python: $(which python)"
        log_message "✓ 当前环境支持vllm推理"
        
        # ========================
        # 测评阶段 - 多机并行推理
        # ========================
        if [ ${PET_NODE_RANK} == 0 ]; then
            log_message "----------------------------------------"
            log_message "开始第 $((iteration + 1)) 轮测评 (多机模式)"
            log_message "----------------------------------------"
            
            # 找到最佳模型检查点
            FINAL_EVAL_LOSS=""
            if [ "$CHECKPOINT_SELECTION" == "latest" ]; then
                # 选择最新创建的checkpoint
                LATEST_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/checkpoint-*/ | head -n 1 | xargs -n 1 basename)
                if [ -z "$LATEST_CHECKPOINT" ]; then
                    log_message "错误: 未找到任何checkpoint目录"
                    exit 1
                fi
                MODEL_CHECKPOINT="$OUTPUT_DIR/$LATEST_CHECKPOINT"
                log_message "使用策略: latest - 最新检查点: $MODEL_CHECKPOINT"
                
                # 从最新checkpoint中提取eval_loss
                if [ -f "$MODEL_CHECKPOINT/trainer_state.json" ]; then
                    FINAL_EVAL_LOSS=$(python3 -c "
import json
import sys
try:
    with open('$MODEL_CHECKPOINT/trainer_state.json', 'r') as f:
        state = json.load(f)
        log_history = state.get('log_history', [])
        # 从后往前找最后一个包含eval_loss的记录
        for entry in reversed(log_history):
            if 'eval_loss' in entry:
                print(entry['eval_loss'])
                sys.exit(0)
        print('N/A')
except:
    print('N/A')
" 2>/dev/null)
                fi
            elif [ "$CHECKPOINT_SELECTION" == "best_eval_loss" ]; then
                # 选择eval_loss最低的checkpoint
                BEST_CHECKPOINT=""
                BEST_EVAL_LOSS="inf"
                
                for checkpoint in "$OUTPUT_DIR"/checkpoint-*/; do
                    if [ ! -d "$checkpoint" ]; then
                        continue
                    fi
                    
                    # 尝试从trainer_state.json读取eval_loss
                    if [ -f "$checkpoint/trainer_state.json" ]; then
                        # 使用Python解析JSON获取最后的eval_loss
                        eval_loss=$(python3 -c "
import json
import sys
try:
    with open('$checkpoint/trainer_state.json', 'r') as f:
        state = json.load(f)
        log_history = state.get('log_history', [])
        # 从后往前找最后一个包含eval_loss的记录
        for entry in reversed(log_history):
            if 'eval_loss' in entry:
                print(entry['eval_loss'])
                sys.exit(0)
        print('inf')
except:
    print('inf')
" 2>/dev/null)
                        
                        if [ "$eval_loss" != "inf" ] && [ -n "$eval_loss" ]; then
                            log_message "检查点: $(basename $checkpoint), eval_loss: $eval_loss"
                            # 比较eval_loss (使用Python进行浮点数比较)
                            is_better=$(python3 -c "
import sys
try:
    current = float('$eval_loss')
    best = float('$BEST_EVAL_LOSS') if '$BEST_EVAL_LOSS' != 'inf' else float('inf')
    print('1' if current < best else '0')
except:
    print('0')
" 2>/dev/null)
                            
                            if [ "$is_better" == "1" ]; then
                                BEST_CHECKPOINT=$(basename "$checkpoint")
                                BEST_EVAL_LOSS="$eval_loss"
                            fi
                        fi
                    fi
                done
                
                if [ -z "$BEST_CHECKPOINT" ]; then
                    # 如果没有找到eval_loss，fallback到使用最新的checkpoint
                    log_message "警告: 未找到eval_loss信息，使用最新检查点"
                    BEST_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/checkpoint-*/ | head -n 1 | xargs -n 1 basename)
                    FINAL_EVAL_LOSS="N/A"
                else
                    FINAL_EVAL_LOSS="$BEST_EVAL_LOSS"
                fi
                
                MODEL_CHECKPOINT="$OUTPUT_DIR/$BEST_CHECKPOINT"
                log_message "使用策略: best_eval_loss - 最佳检查点: $MODEL_CHECKPOINT (eval_loss: $BEST_EVAL_LOSS)"
            else
                log_message "错误: 未知的CHECKPOINT_SELECTION值: $CHECKPOINT_SELECTION"
                exit 1
            fi
            
            # 记录eval_loss到单独的日志文件中
            EVAL_LOSS_LOG_FILE="${BASE_DIR}/eval_loss.log"
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            if [ -n "$FINAL_EVAL_LOSS" ] && [ "$FINAL_EVAL_LOSS" != "N/A" ]; then
                echo "[${TIMESTAMP}] Iteration ${iteration}: eval_loss = ${FINAL_EVAL_LOSS}" >> "$EVAL_LOSS_LOG_FILE"
                log_message "✓ 已记录 eval_loss 到: $EVAL_LOSS_LOG_FILE (eval_loss: ${FINAL_EVAL_LOSS})"
            else
                echo "[${TIMESTAMP}] Iteration ${iteration}: eval_loss = N/A (未找到)" >> "$EVAL_LOSS_LOG_FILE"
                log_message "警告: 未找到 eval_loss 信息，已记录到: $EVAL_LOSS_LOG_FILE"
            fi
            
            # 保存checkpoint路径供worker节点使用
            echo "$MODEL_CHECKPOINT" > "${ITERATION_DIR}/log/current_checkpoint.txt"
            log_message "已保存checkpoint路径: $MODEL_CHECKPOINT"
        fi
        
        # ========================
        # 多机并行推理 - 使用通用函数
        # ========================
        # 调用多机测评函数进行plan_test测评
        multi_node_evaluate "${ITERATION_DIR}/plan_test" "$RESULT_DIR" "test.py" "plan"
        
        # ========================
        # 解析测评结果 - 仅rank0执行
        # ========================
        if [ ${PET_NODE_RANK} -eq 0 ]; then
            # 解析测评结果
            if [ -d "$RESULT_DIR" ]; then
                LATEST_SUMMARY=$(ls -t "$RESULT_DIR"summary_report_*.txt | head -n 1)
                
                if [ -f "$LATEST_SUMMARY" ]; then
                    # 提取测评分数
                    EXACT_SCORE=$(grep "Overall Exact Match Average:" "$LATEST_SUMMARY" | awk '{print $NF}')
                    STEP_SCORE=$(grep "Overall Step Match Average:" "$LATEST_SUMMARY" | awk '{print $NF}')
                    
                    log_message "测评结果 - Exact Match: $EXACT_SCORE, Step Match: $STEP_SCORE"
                    
                    # 保存准确率结果
                    echo "第 $((iteration + 1)) 轮准确率 - Exact Match: $EXACT_SCORE, Step Match: $STEP_SCORE" >> "${ITERATION_DIR}/log/accuracy_log.txt"
                else
                    log_message "警告: 未找到测评结果文件"
                    EXACT_SCORE=0.0
                    STEP_SCORE=0.0
                fi
            else
                log_message "警告: 测评结果目录不存在"
                EXACT_SCORE=0.0
                STEP_SCORE=0.0
            fi
            
            # ========================
            # 错误分析阶段
            # ========================
            log_message "----------------------------------------"
            log_message "开始错误分析"
            log_message "----------------------------------------"
            
            # 调用独立的错误分析脚本
            log_message "分析错误条数..."
            ERROR_ANALYSIS_LOG="${ITERATION_DIR}/log/error_analysis_iteration_${iteration}.log"
            python "${EVAL_DIR}/error_analysis.py" \
                "${RESULT_DIR}" \
                "${ITERATION_DIR}/log" \
                "${iteration}" \
                > "${ERROR_ANALYSIS_LOG}" 2>&1
            
            # 检查错误分析是否成功完成
            if [ $? -eq 0 ]; then
                log_message "✓ 错误分析完成，详细日志已保存到: ${ERROR_ANALYSIS_LOG}"
            else
                log_message "⚠ 错误分析过程中出现错误，请检查日志: ${ERROR_ANALYSIS_LOG}"
            fi
            
            log_message "✓ 错误分析完成"
        fi
        
        # ========================
        # 其余多机并行推理 - 所有节点执行
        # ========================
        # 真实测试集测评
        multi_node_evaluate "${BASE_DIR}/real_test_data" "${ITERATION_DIR}/real_result/" "test.py" "真实测试集"
        
        # BBox测评
        multi_node_evaluate "${ITERATION_DIR}/bbox_test" "${ITERATION_DIR}/bbox_result/" "test_bbox.py" "BBox"
        
        # Keypage测评
        multi_node_evaluate "${ITERATION_DIR}/keypage_test" "${ITERATION_DIR}/keypage_result/" "test_keypage.py" "Keypage"
        
        # 闭环测评
        multi_node_evaluate "${ITERATION_DIR}/close_loop_test" "${ITERATION_DIR}/close_loop_result/" "test_close_loop.py" "闭环"
        
        # 真实BBox测评
        multi_node_evaluate "${BASE_DIR}/real_bbox_test_data" "${ITERATION_DIR}/real_bbox_result/" "test_bbox.py" "真实BBox"
        
        # 真实Keypage测评
        multi_node_evaluate "${BASE_DIR}/real_keypage_test_data" "${ITERATION_DIR}/real_keypage_result/" "test_keypage.py" "真实Keypage"
        
        # 真实闭环测评
        multi_node_evaluate "${BASE_DIR}/real_close_loop_test_data" "${ITERATION_DIR}/real_close_loop_result/" "test_close_loop.py" "真实闭环"
        
        # ========================
        # 数据增强阶段 - 仅rank0执行
        # ========================
        if [ ${PET_NODE_RANK} -eq 0 ]; then
            # 定义结果目录变量
            REAL_TEST_RESULT_DIR="${ITERATION_DIR}/real_result/"
            BBOX_RESULT_DIR="${ITERATION_DIR}/bbox_result/"
            KEYPAGE_RESULT_DIR="${ITERATION_DIR}/keypage_result/"
            CLOSE_LOOP_RESULT_DIR="${ITERATION_DIR}/close_loop_result/"
            REAL_BBOX_RESULT_DIR="${ITERATION_DIR}/real_bbox_result/"
            REAL_KEYPAGE_RESULT_DIR="${ITERATION_DIR}/real_keypage_result/"
            REAL_CLOSE_LOOP_RESULT_DIR="${ITERATION_DIR}/real_close_loop_result/"
            
            # ========================
            # Prepare next iteration (if any) + optional data enhancement
            # ========================
            NEXT_ROUND=$((iteration + 1))
            if [ "$NEXT_ROUND" -lt "$MAX_ITERATIONS" ]; then
                NEXT_ITERATION_DIR="${BASE_DIR}/iteration_${NEXT_ROUND}"
                log_message "准备下一轮目录: ${NEXT_ITERATION_DIR}"
                mkdir -p "${NEXT_ITERATION_DIR}/Train_data"
                mkdir -p "${NEXT_ITERATION_DIR}/Test_data"
                mkdir -p "${NEXT_ITERATION_DIR}/plan_test"
                mkdir -p "${NEXT_ITERATION_DIR}/bbox_test"
                mkdir -p "${NEXT_ITERATION_DIR}/keypage_test"
                mkdir -p "${NEXT_ITERATION_DIR}/close_loop_test"
                mkdir -p "${NEXT_ITERATION_DIR}/model"
                mkdir -p "${NEXT_ITERATION_DIR}/log"
                mkdir -p "${NEXT_ITERATION_DIR}/result"

                # Carry forward datasets
                [ -d "${ITERATION_DIR}/Train_data" ] && cp -r "${ITERATION_DIR}/Train_data"/* "${NEXT_ITERATION_DIR}/Train_data/" 2>/dev/null || true
                [ -d "${ITERATION_DIR}/Test_data" ] && cp -r "${ITERATION_DIR}/Test_data"/* "${NEXT_ITERATION_DIR}/Test_data/" 2>/dev/null || true
                for sub in plan_test bbox_test keypage_test close_loop_test; do
                    if [ -d "${ITERATION_DIR}/${sub}" ] && [ -n "$(ls -A "${ITERATION_DIR}/${sub}" 2>/dev/null)" ]; then
                        cp -r "${ITERATION_DIR}/${sub}"/* "${NEXT_ITERATION_DIR}/${sub}/" 2>/dev/null || true
                    fi
                done
                for enhancement_dir in "${ITERATION_DIR}"/data_enhancement*; do
                    if [ -d "$enhancement_dir" ] && [ -n "$(ls -A "$enhancement_dir" 2>/dev/null)" ]; then
                        enhancement_name=$(basename "$enhancement_dir")
                        mkdir -p "${NEXT_ITERATION_DIR}/${enhancement_name}"
                        cp -r "$enhancement_dir"/* "${NEXT_ITERATION_DIR}/${enhancement_name}/" 2>/dev/null || true
                    fi
                done

                if [[ "${ENABLE_DATA_ENHANCEMENT}" == "1" ]]; then
                    log_message "开始轮间数据增强 (ENABLE_DATA_ENHANCEMENT=1)"
                    RESULT_DIR_PLAN="${ITERATION_DIR}/result/"
                    BBOX_RESULT_DIR="${ITERATION_DIR}/bbox_result/"
                    KEYPAGE_RESULT_DIR="${ITERATION_DIR}/keypage_result/"
                    CLOSE_LOOP_RESULT_DIR="${ITERATION_DIR}/close_loop_result/"

                    if [ -d "${RESULT_DIR_PLAN}" ]; then
                        python "${ENHANCE_DIR}/data_enhancement.py" \
                            "${ITERATION_DIR}/log" \
                            "${NEXT_ITERATION_DIR}/Train_data" \
                            "${RESULT_DIR_PLAN}" \
                            "${iteration}" \
                            > "${ITERATION_DIR}/log/enhancement_iteration_${iteration}.log" 2>&1 || \
                            log_message "⚠ plan 数据增强失败，详见 log"
                    fi
                    if [ -d "${BBOX_RESULT_DIR}" ] && [ -n "$(ls -A "${BBOX_RESULT_DIR}" 2>/dev/null)" ]; then
                        python "${ENHANCE_DIR}/data_enhancement_bbox.py" \
                            "${ITERATION_DIR}/log" \
                            "${NEXT_ITERATION_DIR}/Train_data" \
                            "${BBOX_RESULT_DIR}" \
                            "${iteration}" \
                            > "${ITERATION_DIR}/log/bbox_enhancement_iteration_${iteration}.log" 2>&1 || true
                    fi
                    if [ -d "${KEYPAGE_RESULT_DIR}" ] && [ -n "$(ls -A "${KEYPAGE_RESULT_DIR}" 2>/dev/null)" ]; then
                        python "${ENHANCE_DIR}/data_enhancement_keypage.py" \
                            "${ITERATION_DIR}/log" \
                            "${NEXT_ITERATION_DIR}/Train_data" \
                            "${KEYPAGE_RESULT_DIR}" \
                            "${iteration}" \
                            > "${ITERATION_DIR}/log/keypage_enhancement_iteration_${iteration}.log" 2>&1 || true
                    fi
                    if [ -d "${CLOSE_LOOP_RESULT_DIR}" ] && [ -n "$(ls -A "${CLOSE_LOOP_RESULT_DIR}" 2>/dev/null)" ]; then
                        python "${ENHANCE_DIR}/data_enhancement_close_loop.py" \
                            "${ITERATION_DIR}/log" \
                            "${NEXT_ITERATION_DIR}/Train_data" \
                            "${CLOSE_LOOP_RESULT_DIR}" \
                            "${iteration}" \
                            > "${ITERATION_DIR}/log/close_loop_enhancement_iteration_${iteration}.log" 2>&1 || true
                    fi
                    log_message "✓ 轮间数据增强完成"
                else
                    log_message "跳过轮间数据增强 (ENABLE_DATA_ENHANCEMENT=${ENABLE_DATA_ENHANCEMENT})"
                fi
            else
                log_message "已是最后一轮 (iteration=${iteration}, MAX_ITERATIONS=${MAX_ITERATIONS})，不准备下一轮目录"
            fi

            
            # ========================
            # 退出conda环境，回到基础pip环境准备下一轮训练
            # ========================
            log_message "=========================================="
            log_message "退出conda环境，回到基础pip环境（训练环境）"
            log_message "=========================================="
            conda deactivate
            log_message "✓ 当前Python: $(which python)"
            log_message "✓ 已回到基础环境，准备下一轮训练"
        fi
        
        # ========================
        # 退出conda环境（所有节点）
        # ========================
        if [ ${PET_NODE_RANK} -ne 0 ]; then
            conda deactivate
            log_message "✓ Worker节点退出conda环境"
        fi
        
        iteration=$((iteration + 1))
        log_message "=========================================="
        log_message "Iteration $((iteration - 1)) 完成"
        log_message "=========================================="
    done
    
    log_message "================================================"
    log_message "配置 ${config} 处理完成，总共执行了 $iteration 轮训练"
    log_message "================================================"
done

log_message "=========================================="
log_message "所有配置处理完成！"
log_message "主日志文件: ${MAIN_LOG_FILE}"
log_message "=========================================="