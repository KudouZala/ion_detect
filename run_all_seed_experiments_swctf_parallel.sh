#!/bin/bash
# SWC-TF 并行版：按 seed 并发运行，每个 seed 内仍保持
# SWC-PSWM -> Exp-A -> Exp-B -> Exp-C -> Exp-D -> (SVM/LSTM/Transformer) 的顺序。
#
# 用法：
#   MAX_PARALLEL_SEEDS=2 bash run_all_seed_experiments_swctf_parallel.sh >& run_all_seed_experiments_swctf_parallel.log

cd "$(dirname "$0")"

trap 'echo "中断信号收到，正在停止..."; kill 0; exit 1' INT QUIT TERM

if [ -f "/home/kudouzala/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/kudouzala/miniconda3/etc/profile.d/conda.sh"
    conda activate ion_detect
fi

CFG_DIR="scripts/machine_learning_code/configs/paper/seeds_swctf"
LOG_DIR="scripts/machine_learning_code/logs_20260323_3/seed_runs_swctf"
MAIN_SCRIPT="scripts/machine_learning_code/main_plus.py"
SVM_SCRIPT="scripts/machine_learning_code/main_svm.py"
LSTM_SCRIPT="scripts/machine_learning_code/main_lstm.py"
TRF_SCRIPT="scripts/machine_learning_code/main_transformer.py"
BASELINE_EPOCHS=1000
BASELINE_EVAL_EVERY=10
MAX_PARALLEL_SEEDS=${MAX_PARALLEL_SEEDS:-2}

mkdir -p "${LOG_DIR}"

SEEDS=(123 2025 3407 27182 31415)
VARIANTS=(
    SWC-PSWM_plus_swctf
    exp_a_plus_swctf
    exp_b_plus_swctf
    exp_c_plus_swctf
    exp_d_plus_swctf
)

run_one_seed() {
    local seed="$1"
    local seed_log="${LOG_DIR}/seed_${seed}_parallel.log"
    : > "${seed_log}"

    {
        echo "======================================================================"
        echo "Seed ${seed} 开始: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "======================================================================"

        for variant in "${VARIANTS[@]}"; do
            local cfg_name="${variant}_seed${seed}"
            local cfg_path="${CFG_DIR}/${cfg_name}.yaml"
            local log_path="${LOG_DIR}/${cfg_name}.log"

            echo "--------------------------------------------------------------------"
            echo "开始运行: ${cfg_name}"
            echo "配置: ${cfg_path}"
            echo "日志: ${log_path}"
            echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "--------------------------------------------------------------------"

            python "${MAIN_SCRIPT}" --config "${cfg_path}" --train > "${log_path}" 2>&1
            local exit_code=$?
            if [ ${exit_code} -ne 0 ]; then
                echo "❌ ${cfg_name} 运行失败 (exit code: ${exit_code})"
                echo "查看日志: ${log_path}"
            else
                echo "✅ ${cfg_name} 运行完成"
            fi
            echo ""
            sleep 5
        done

        local expd_cfg="${CFG_DIR}/exp_d_plus_swctf_seed${seed}.yaml"
        local svm_log="${LOG_DIR}/svm_seed${seed}.log"
        local lstm_log="${LOG_DIR}/lstm_seed${seed}.log"
        local trf_log="${LOG_DIR}/transformer_seed${seed}.log"

        echo "~~~~~~~~~~~~~~~~ 基线模型开始 (seed=${seed}, 同一数据划分) ~~~~~~~~~~~~~~~~"

        python "${SVM_SCRIPT}" \
            --config "${expd_cfg}" \
            --out_csv "${LOG_DIR}/svm_seed${seed}_details.csv" > "${svm_log}" 2>&1
        local svm_exit=$?
        if [ ${svm_exit} -ne 0 ]; then
            echo "❌ svm_seed${seed} 运行失败 (exit code: ${svm_exit})"
            echo "查看日志: ${svm_log}"
        else
            echo "✅ svm_seed${seed} 运行完成"
        fi

        python "${LSTM_SCRIPT}" \
            --config "${expd_cfg}" \
            --epochs ${BASELINE_EPOCHS} \
            --eval_every ${BASELINE_EVAL_EVERY} \
            --out_csv "${LOG_DIR}/lstm_seed${seed}_details.csv" > "${lstm_log}" 2>&1
        local lstm_exit=$?
        if [ ${lstm_exit} -ne 0 ]; then
            echo "❌ lstm_seed${seed} 运行失败 (exit code: ${lstm_exit})"
            echo "查看日志: ${lstm_log}"
        else
            echo "✅ lstm_seed${seed} 运行完成"
        fi

        python "${TRF_SCRIPT}" \
            --config "${expd_cfg}" \
            --epochs ${BASELINE_EPOCHS} \
            --eval_every ${BASELINE_EVAL_EVERY} \
            --out_csv "${LOG_DIR}/transformer_seed${seed}_details.csv" > "${trf_log}" 2>&1
        local trf_exit=$?
        if [ ${trf_exit} -ne 0 ]; then
            echo "❌ transformer_seed${seed} 运行失败 (exit code: ${trf_exit})"
            echo "查看日志: ${trf_log}"
        else
            echo "✅ transformer_seed${seed} 运行完成"
        fi

        echo "~~~~~~~~~~~~~~~~ 基线模型结束 (seed=${seed}) ~~~~~~~~~~~~~~~~"
        echo "Seed ${seed} 完成: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    } >> "${seed_log}" 2>&1
}

echo "并行 seed 数: ${MAX_PARALLEL_SEEDS}"
echo "日志目录: ${LOG_DIR}"

pids=()
running=0
for seed in "${SEEDS[@]}"; do
    run_one_seed "${seed}" &
    pids+=($!)
    running=$((running + 1))
    echo "已启动 seed=${seed} (pid=${pids[-1]})"

    if [ "${running}" -ge "${MAX_PARALLEL_SEEDS}" ]; then
        wait -n
        running=$((running - 1))
    fi
done

wait
echo "全部并行任务完成: $(date '+%Y-%m-%d %H:%M:%S')"
