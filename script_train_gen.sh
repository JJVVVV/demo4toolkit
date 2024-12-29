#!/bin/bash

# ./script_train_gen.sh
# nohup ./script_train_gen.sh &
# tmux new-session -d -s train_task 'bash ./script_train_gen.sh > /dev/null 2>&1'

# PATH="/home/qwe/miniconda3/envs/jjvv/bin:$PATH"

if [ -z "$1" ]; then
  CUDA_VISIBLE_DEVICES=1
else
  CUDA_VISIBLE_DEVICES=$1
fi

# ###################################parameters#########################################
deepspeed_config_file="./configs/ds_zero2.hjson"
activation_checkpointing=True
accumulate_step=1
cut_input_from_output=None


# 模型与任务
model_structure="decoder"
task_type="generate"
dashboard="None"
dataset_name=fool_gen
text_type='ORI'

parts=("all")

model_dir=None
model_type="Llama-3.2-1B-Instruct"
# model_type="Meta-Llama-3.1-8B-Instruct"

model_names=("baseline-lora")

# 训练参数
seeds=(2)

# torch_dtype="float32"
torch_dtype="auto"
fp16=False
bf16=False

test_in_epoch=True

opt_type=AdamW
batch_size=2
batch_size_infer=2
epochs=3
max_length=None
learning_rate='2e-5'
weight_decay=0.1
metric='rougeL'
max_new_tokens=10
padding_side="None"
sch_type="WarmupDecayLR"
warmup_ratio=0.1



# 数据集文件
if [ "$text_type" = "ORI" ]; then
    train_file_path="data/fool/train/train_generate.json"
    val_file_path="data/fool/dev/dev_generate.json"
    test_file_path=None
elif [ "$text_type" = "XXX" ]; then
    exit 1
    :
fi


# ###################################parameters#########################################


# 定义一个数组，存放可用cuda
# IFS=',' cudas=($CUDA_VISIBLE_DEVICES) IFS=' '
IFS='/' cudas=($CUDA_VISIBLE_DEVICES) IFS=' '
# 计算每个每个任务可用cuda数量
IFS=',' nproc_pre_node=(${cudas[0]}) IFS=' '
nproc_pre_node=${#nproc_pre_node[@]}
# 定义一个变量，表示最大并行数
parallel=${#cudas[@]}
# 定义一个数组，存放当前运行的进程号
pids=()
# 定义一个字典, 记录PID运行在哪个CUDA设备上
declare -A pid_cuda


for part in ${parts[@]}
do
    # 遍历所有的种子
    for model_name in ${model_names[@]}
    do
        for seed in ${seeds[@]}
        do
            # 判断有无console目录, 没有则创建
            log_file="logs/$dataset_name-$text_type-$model_type-$model_name-$epochs-$batch_size-$learning_rate-$seed.ansi.log"
            log_dir=${log_file%/*}
            if [ ! -d log_dir ]; then
                mkdir -p $log_dir
            fi
            # 如果当前运行的进程数达到最大并行数，就等待任意一个进程结束: 从数组pids中删除结束进程的PID, 释放一个CUDA
            if [ ${#pids[@]} -eq $parallel ]; then
                wait -n ${pids[@]}
                # 删除已经结束的进程号, 释放一个可用的cuda
                for pid in ${pids[@]}
                do
                if ! ps -p $pid > /dev/null ; then
                    # echo $pid
                    finishedPID=$pid
                    break
                fi
                done
                echo "finishPID: $finishedPID"
                pids=(${pids[@]/$finishedPID/})
                cudas+=(${pid_cuda[$finishedPID]})
                echo "freeCUDA: ${pid_cuda[$finishedPID]}"
                unset pid_cuda[$finishedPID]
                echo "runningProcesses: ${pids[@]}"
                echo "avaliableCUDAs: ${cudas[@]}"
                echo
            fi
            # 启动一个新训练任务: 使用一个可用的cuda,并把它的PID添加到数组pids中
            cuda=${cudas[0]}
            unset cudas[0]
            cudas=(${cudas[@]})

            # ###################################训练程序#########################################
            # TORCH_DISTRIBUTED_DEBUG=INFO \
            if [ $nproc_pre_node -gt 1 ]; then
                executable="torchrun \
                --rdzv-backend=c10d \
                --rdzv-endpoint=localhost:0 \
                --nnodes=1 \
                --nproc-per-node=$nproc_pre_node \
                --master-port 29503"
                parallel_mode="deepspeed"
            else
                executable="python"
                parallel_mode="None"
                deepspeed_config_file="None"
            fi
            CUDA_VISIBLE_DEVICES=$cuda \
            $executable ./training.py \
                --dataset_name $dataset_name \
                --num_classification $num_classification \
                --model_type $model_type \
                --model_name $model_name \
                --train_file_path $train_file_path \
                --val_file_path $val_file_path \
                --test_file_path $test_file_path \
                --seed $seed \
                --opt_type $opt_type \
                --sch_type $sch_type \
                --opt_lr $learning_rate \
                --epochs $epochs \
                --train_batch_size $batch_size \
                --infer_batch_size $batch_size_infer \
                --sch_warmup_ratio_steps $warmup_ratio \
                --max_length $max_length \
                --metric $metric \
                --eval_every_half_epoch $test_in_epoch \
                --gradient_accumulation_steps $accumulate_step \
                --fp16 $fp16 \
                --bf16 $bf16 \
                --opt_weight_decay $weight_decay \
                --dashboard $dashboard \
                --text_type $text_type \
                --part $part \
                --model_dir $model_dir \
                --model_structure $model_structure \
                --task_type $task_type \
                --cut_input_from_output $cut_input_from_output \
                --torch_dtype $torch_dtype \
                --max_new_tokens $max_new_tokens \
                --save_last_ckpt False \
                --cache_dataset $cache_dataset \
                --activation_checkpointing $activation_checkpointing \
                --padding_side $padding_side \
                --deepspeed_config $deepspeed_config_file \
                --parallel_mode $parallel_mode \
                --padding_to_max_length False \
                # > $log_file 2>&1 &

            # ###################################训练程序#########################################
            newPID=$!
            pids+=($newPID)
            pid_cuda[$newPID]=$cuda
            echo "newPID: $newPID"
            echo "useCUDA: ${pid_cuda[$newPID]}"
            echo "runningProcesses: ${pids[@]}"
            echo "avaliableCUDAs: ${cudas[@]}"
            echo

            # while [ ! -f "console/seed-$seed.log" ]; do
            #   echo "waiting trainScript.sh to run in the background."
            #   sleep 1
        done
    done
done

# 等待所有剩余的进程结束
wait ${pids[@]}