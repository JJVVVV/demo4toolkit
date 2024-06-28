#!/bin/bash

# nohup ./script_train_gen.sh > /dev/null 2>&1 &
# tmux new-session -d -s train_task 'bash ./script_train_gen.sh > /dev/null 2>&1'

PATH="/home/qwe/miniconda3/envs/jjvv/bin:$PATH"

CUDA_VISIBLE_DEVICES=1

# 模型与任务
dataset_name=fool
model_type="deepseek-coder-1.3b-instruct"
model_dir="/home/qwe/test/pretrained_model/$model_type"
if [[ "$model_type" == *"deepseek"* ]]; then
    model_type="deepseek-ai/$model_type"
else
    model_type="meta-llama/$model_type"
fi
echo $model_type
model_structure=decoder
task_type=generate


# 训练参数
model_name="baseline"
text_type=ORI
epochs=3
train_batch_size=32
infer_batch_size=16
max_length=2048
gradient_accumulation_steps=1
opt_lrs=("2e-5")
opt_weight_decay=0.01
sch_type=WarmupDecayLR
sch_warmup_ratio_steps=0.1
metric='rougeL'
activation_checkpointing=True

if [ "$model_type" = "deepseek-ai/deepseek-coder-1.3b-instruct" ]; then
    fp16=False
    bf16=True
    torch_dtype=bfloat16
    deepspeed_config_file=./configs/ds_zero2.hjson
    hf_gen_config_file="./configs/generate_config.json"
    gradient_accumulation_steps=1
elif [ "$model_type" = "XXX" ]; then
    exit 1
fi


# 数据集文件
if [ "$text_type" = "ORI" ]; then
    train_file_path="data/$dataset_name/train/train_generate.json"
    val_file_path="data/$dataset_name/dev/dev_generate.json"
    test_file_path=None
elif [ "$text_type" = "XXX" ]; then
    exit 1
    :
fi


IFS=',' nproc_per_node=($CUDA_VISIBLE_DEVICES) IFS=' '
nproc_per_node=${#nproc_per_node[@]}
echo $nproc_per_node


for opt_lr in "${opt_lrs[@]}"
do
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    torchrun --nnodes 1 --nproc_per_node $nproc_per_node --master-port 29503 train.py \
        --model_structure $model_structure \
        --task_type $task_type \
        --parallel_mode "deepspeed" \
        --model_name $model_name \
        --model_type $model_type \
        --dataset_name $dataset_name \
        --torch_dtype $torch_dtype \
        --max_length $max_length \
        --dashboard "None" \
        --metric $metric \
        --model_dir ${model_dir} \
        --train_file_path ${train_file_path} \
        --val_file_path ${val_file_path} \
        --test_file_path $test_file_path \
        --train_batch_size ${train_batch_size} \
        --infer_batch_size ${infer_batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --seed 0 \
        --fp16 $fp16 \
        --bf16 $bf16 \
        --epochs $epochs \
        --opt_type "AdamW" \
        --opt_lr ${opt_lr} \
        --sch_type $sch_type \
        --sch_warmup_ratio_steps $sch_warmup_ratio_steps \
        --opt_weight_decay $opt_weight_decay \
        --ddp_timeout 30000 \
        --logging_steps 1 \
        --padding_side "left" \
        --save_dir None \
        --cut_input_from_output True \
        --hf_gen_config_file $hf_gen_config_file \
        --use_deepspeed_ckpt False \
        --save_ckpts False \
        --save_all_ckpts False \
        --save_last_ckpt False \
        --cache_dataset False \
        --deepspeed_config $deepspeed_config_file \
        --padding_to_max_length False \
        --text_type $text_type \
        --activation_checkpointing $activation_checkpointing \
        > training.log 2>&1
done
