#!/bin/bash

# ./script_train_cla.sh
# nohup ./script_train_cla.sh &
# tmux new-session -d -s train_task 'bash ./script_train_cla.sh > /dev/null 2>&1'

# PATH="/home/qwe/miniconda3/envs/jjvv/bin:$PATH"

if [ -z "$1" ]; then
  CUDA_VISIBLE_DEVICES=0
else
  CUDA_VISIBLE_DEVICES=$1
fi


# 模型与任务
dashboard=None
dataset_name=fool
num_classification=2
model_dir=None
model_type="bert-base-chinese"
model_structure="encoder"
task_type=classify

text_type=ORI
model_name="baseline"
part=all

# 训练参数
parallel_mode="deepspeed"
parallel_mode=None

activation_checkpointing=False
gradient_accumulation_steps=1

seeds=(2)
epochs=3
train_batch_size=16
infer_batch_size=16
max_length_input=None
opt_type="AdamW"
opt_lrs=("2e-5")
opt_weight_decay=0.01
sch_type=WarmupDecayLR
sch_warmup_ratio_steps=0.1
metric='accuracy'
eval_every_half_epoch=False

if [ "$model_type" = "bert-base-chinese" ]; then
    fp16=True
    bf16=False
    torch_dtype=auto
    deepspeed_config_file=./configs/ds_zero2.hjson
    # hf_generation_config_file="./configs/generate_config.json"
    gradient_accumulation_steps=1
elif [ "$model_type" = "XXX" ]; then
    exit 1
    :
fi

cache_dataset=False
# 数据集文件
if [ "$text_type" = "ORI" ]; then
    train_file_path="data/fool/train/train_classify.json"
    # train_file_path="data/fool/train/train_classify.txt"
    val_file_path="data/fool/dev/dev_classify.json"
    test_file_path=None
elif [ "$text_type" = "XXX" ]; then
    exit 1
    :
fi



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

    # 遍历所有的种子
for seed in ${seeds[@]}
do
    for opt_lr in ${opt_lrs[@]}
    do
        # 判断有无console目录, 没有则创建
        log_file="logs/$dataset_name-$text_type-$model_type-$model_name-$epochs-$train_batch_size-$opt_lr-$part-$seed.ansi.log"
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
        HF_ENDPOINT="https://hf-mirror.com" CUDA_VISIBLE_DEVICES=$cuda \
        $executable ./train.py \
            --dataset_name $dataset_name \
            --model_type $model_type \
            --model_name $model_name \
            --train_file_path $train_file_path \
            --val_file_path $val_file_path \
            --test_file_path $test_file_path \
            --seed $seed \
            --opt_type $opt_type \
            --sch_type $sch_type \
            --opt_lr $opt_lr \
            --epochs $epochs \
            --train_batch_size $train_batch_size \
            --infer_batch_size $infer_batch_size \
            --sch_warmup_ratio_steps $sch_warmup_ratio_steps \
            --max_length_input $max_length_input \
            --metric $metric \
            --eval_every_half_epoch $eval_every_half_epoch \
            --gradient_accumulation_steps $gradient_accumulation_steps \
            --fp16 $fp16 \
            --bf16 $bf16 \
            --opt_weight_decay $opt_weight_decay \
            --dashboard $dashboard \
            --save_ckpts False \
            --save_all_ckpts False \
            --model_dir $model_dir \
            --model_structure $model_structure \
            --task_type $task_type \
            --cache_dataset $cache_dataset \
            --activation_checkpointing $activation_checkpointing \
            --padding_side None \
            --cut_input_from_output None \
            --parallel_mode $parallel_mode \
            --padding_to_max_length False \
            --logging_steps 1 \
            --save_dir None \
            --use_deepspeed_ckpt False \
            --deepspeed_config $deepspeed_config_file \
            --num_classification $num_classification \
            --text_type $text_type \
            --part $part \
            --torch_dtype $torch_dtype \
            --ddp_timeout 30000 \
            # --hf_generation_config_file $hf_generation_config_file \
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
    done
done

# 等待所有剩余的进程结束
wait ${pids[@]}