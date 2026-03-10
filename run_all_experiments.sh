#!/bin/bash
# 依次运行所有消融实验，每个实验完成后再启动下一个
#
# 用法：
#   ./run_all_experiments.sh >& run_all.log
#
# 查看总进度：  tail -f run_all.log
# 查看某个实验：tail -f exp_a.log

cd "$(dirname "$0")"

# Ctrl+C 时同时杀掉正在运行的 python 子进程
trap 'echo "中断信号收到，正在停止..."; kill 0; exit 1' INT QUIT TERM

PYCACHE_DIR="scripts/machine_learning_code/__pycache__"
CFG_PAPER_DIR="scripts/machine_learning_code/configs/paper"

CONFIGS=(
    # 论文主模型 + 论文消融
    SWC-PSWM
    exp_a
    exp_b
    exp_c
    exp_d
)

TOTAL=${#CONFIGS[@]}

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    idx=$((i + 1))
    echo "============================================================"
    echo "[${idx}/${TOTAL}] 开始运行: ${cfg}"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    rm -rf "${PYCACHE_DIR}"
    echo "  已清理 __pycache__"

    cfg_path="${CFG_PAPER_DIR}/${cfg}.yaml"

    python scripts/machine_learning_code/main.py \
        --config "${cfg_path}" \
        --train >& "${cfg}.log"

    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  ❌ ${cfg} 运行失败 (exit code: ${exit_code})，跳过继续下一个"
        echo "  查看日志: ${cfg}.log"
    else
        echo "  ✅ ${cfg} 运行完成"
    fi

    echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    sleep 10
done

echo "============================================================"
echo "全部 ${TOTAL} 个实验已完成！ $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
