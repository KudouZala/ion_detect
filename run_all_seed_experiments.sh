#!/bin/bash
# 按随机种子分组，依次运行：
# seed_1: SWC-PSWM -> Exp-A -> Exp-B -> Exp-C -> Exp-D
# seed_2: SWC-PSWM -> Exp-A -> Exp-B -> Exp-C -> Exp-D
# ...
#
# 用法：
#   bash run_all_seed_experiments.sh >& run_all_seed_experiments.log

cd "$(dirname "$0")"

trap 'echo "中断信号收到，正在停止..."; kill 0; exit 1' INT QUIT TERM

if [ -f "/home/kudouzala/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/kudouzala/miniconda3/etc/profile.d/conda.sh"
    conda activate ion_detect
fi

PYCACHE_DIR="scripts/machine_learning_code/__pycache__"
CFG_DIR="scripts/machine_learning_code/configs/paper/seeds"
LOG_DIR="scripts/machine_learning_code/logs_20260323_3/seed_runs"
MAIN_SCRIPT="scripts/machine_learning_code/main_plus.py"
mkdir -p "${LOG_DIR}"

SEEDS=(123 2025 3407 27182 31415)
VARIANTS=(
    SWC-PSWM_plus
    exp_a_plus
    exp_b_plus
    exp_c_plus
    exp_d_plus
)

TOTAL=$(( ${#SEEDS[@]} * ${#VARIANTS[@]} ))
COUNT=0

for seed in "${SEEDS[@]}"; do
    echo "======================================================================"
    echo "开始随机种子 seed=${seed} 的一组实验"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"

    for variant in "${VARIANTS[@]}"; do
        COUNT=$((COUNT + 1))
        cfg_name="${variant}_seed${seed}"
        cfg_path="${CFG_DIR}/${cfg_name}.yaml"
        log_path="${LOG_DIR}/${cfg_name}.log"

        echo "--------------------------------------------------------------------"
        echo "[${COUNT}/${TOTAL}] 开始运行: ${cfg_name}"
        echo "  配置: ${cfg_path}"
        echo "  日志: ${log_path}"
        echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "--------------------------------------------------------------------"

        rm -rf "${PYCACHE_DIR}"

        python "${MAIN_SCRIPT}" \
            --config "${cfg_path}" \
            --train > "${log_path}" 2>&1

        exit_code=$?
        if [ ${exit_code} -ne 0 ]; then
            echo "  ❌ ${cfg_name} 运行失败 (exit code: ${exit_code})"
            echo "  查看日志: ${log_path}"
        else
            echo "  ✅ ${cfg_name} 运行完成"
        fi

        echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        sleep 10
    done
done

echo "======================================================================"
echo "全部 ${TOTAL} 个 seed 实验已完成！ $(date '+%Y-%m-%d %H:%M:%S')"
echo "日志目录: ${LOG_DIR}"
echo "======================================================================"
