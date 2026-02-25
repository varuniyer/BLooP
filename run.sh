#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
git config --global --add safe.directory /home/user/work/BLooP

case $MODE in

    parallel_sweep)
        num_gpus=$(nvidia-smi -L | wc -l)
        num_procs=${1:-$num_gpus}
        gpus_per_proc=$((num_gpus / num_procs))
        sweep_id=$(wandb sweep sweep_config.yml 2>&1 | tee /dev/stderr | awk 'NR==2 {print \$NF}')
        for proc_id in $(seq 0 $((num_procs - 1))); do
            gpu_ids=$(seq -s, $((proc_id * gpus_per_proc)) $(((proc_id + 1) * gpus_per_proc - 1)) | sed 's/,$//')
            echo "Starting sweep $sweep_id on GPU(s) $gpu_ids"
            CUDA_VISIBLE_DEVICES=$gpu_ids nohup wandb agent $sweep_id > agent_$gpu_ids.log 2>&1 &
        done
        wait
        for proc_id in $(seq 0 $((num_procs - 1))); do
            echo "GPU $proc_id log:"
            cat agent_$proc_id.log
        done
        cat /tmp/debug-cli.user.log
        ;;

    single_run)
        python main.py \
            --model google/gemma-2-9b-it \
            --dataset ccsum \
            --max_input_len 2048 \
            --split test \
            --subsample 1 \
            --beam_width 4 \
            --alpha 6
            # alpha: 3 for llama, 4 for mistral, 6 for gemma
            # beam_width: 12 for llama, 5 for mistral, 4 for gemma
            # max_input_len: 4096 for llama, 64000 for mistral, 2048 for gemma
        ;;

    collect_results)
        python collect_wandb_results.py
        for file in perf_data.csv cache_usage_data.csv promotion_effect_data.csv; do
            echo "$file:"
            cat $file
        done
        ;;

    *)
        echo "Invalid mode: $MODE"
        exit 1
        ;;
esac
