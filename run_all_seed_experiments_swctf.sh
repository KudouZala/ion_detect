#!/bin/bash
# SWC-TF: 按随机种子分组依次运行
# seed_1: SWC-PSWM_swctf -> Exp-A_swctf -> Exp-B_swctf -> Exp-C_swctf -> Exp-D_swctf
#
# 用法：
#   bash run_all_seed_experiments_swctf.sh >& run_all_seed_experiments_swctf.log

cd "$(dirname "$0")"

trap 'echo "中断信号收到，正在停止..."; kill 0; exit 1' INT QUIT TERM

if [ -f "/home/kudouzala/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/kudouzala/miniconda3/etc/profile.d/conda.sh"
    conda activate ion_detect
fi

PYCACHE_DIR="scripts/machine_learning_code/__pycache__"
CFG_DIR="scripts/machine_learning_code/configs/paper/seeds_swctf"
LOG_DIR="scripts/machine_learning_code/logs_20260323_4/seed_runs_swctf"
MAIN_SCRIPT="scripts/machine_learning_code/main_plus.py"
SVM_SCRIPT="scripts/machine_learning_code/main_svm.py"
LSTM_SCRIPT="scripts/machine_learning_code/main_lstm.py"
TRF_SCRIPT="scripts/machine_learning_code/main_transformer.py"
BASELINE_EPOCHS=1000
BASELINE_EVAL_EVERY=10
mkdir -p "${LOG_DIR}"

SEEDS=(123 2025 3407 27182 31415)
VARIANTS=(
    SWC-PSWM_plus_swctf
    exp_a_plus_swctf
    exp_b_plus_swctf
    exp_c_plus_swctf
    exp_d_plus_swctf
)

TOTAL=$(( ${#SEEDS[@]} * ${#VARIANTS[@]} ))
COUNT=0

for seed in "${SEEDS[@]}"; do
    echo "======================================================================"
    echo "开始随机种子 seed=${seed} 的一组 swctf 实验"
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

        # 在 Exp-D 结束后，基于同一 train/test split 继续跑 SVM / LSTM / Transformer
        if [ "${variant}" = "exp_d_plus_swctf" ]; then
            echo "~~~~~~~~~~~~~~~~ 基线模型开始 (seed=${seed}, 同一数据划分) ~~~~~~~~~~~~~~~~"

            svm_log="${LOG_DIR}/svm_seed${seed}.log"
            lstm_log="${LOG_DIR}/lstm_seed${seed}.log"
            trf_log="${LOG_DIR}/transformer_seed${seed}.log"

            python "${SVM_SCRIPT}" \
                --config "${cfg_path}" \
                --out_csv "${LOG_DIR}/svm_seed${seed}_details.csv" > "${svm_log}" 2>&1
            svm_exit=$?
            if [ ${svm_exit} -ne 0 ]; then
                echo "  ❌ svm_seed${seed} 运行失败 (exit code: ${svm_exit})"
                echo "  查看日志: ${svm_log}"
            else
                echo "  ✅ svm_seed${seed} 运行完成"
            fi

            python "${LSTM_SCRIPT}" \
                --config "${cfg_path}" \
                --epochs ${BASELINE_EPOCHS} \
                --eval_every ${BASELINE_EVAL_EVERY} \
                --out_csv "${LOG_DIR}/lstm_seed${seed}_details.csv" > "${lstm_log}" 2>&1
            lstm_exit=$?
            if [ ${lstm_exit} -ne 0 ]; then
                echo "  ❌ lstm_seed${seed} 运行失败 (exit code: ${lstm_exit})"
                echo "  查看日志: ${lstm_log}"
            else
                echo "  ✅ lstm_seed${seed} 运行完成"
            fi

            python "${TRF_SCRIPT}" \
                --config "${cfg_path}" \
                --epochs ${BASELINE_EPOCHS} \
                --eval_every ${BASELINE_EVAL_EVERY} \
                --out_csv "${LOG_DIR}/transformer_seed${seed}_details.csv" > "${trf_log}" 2>&1
            trf_exit=$?
            if [ ${trf_exit} -ne 0 ]; then
                echo "  ❌ transformer_seed${seed} 运行失败 (exit code: ${trf_exit})"
                echo "  查看日志: ${trf_log}"
            else
                echo "  ✅ transformer_seed${seed} 运行完成"
            fi

            echo "~~~~~~~~~~~~~~~~ 基线模型结束 (seed=${seed}) ~~~~~~~~~~~~~~~~"
            echo ""
        fi
        sleep 10
    done
done

echo "======================================================================"
echo "全部 ${TOTAL} 个 swctf seed 实验已完成！ $(date '+%Y-%m-%d %H:%M:%S')"
echo "日志目录: ${LOG_DIR}"
echo "======================================================================"
