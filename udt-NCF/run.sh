#!/bin/bash

# Configurations
dataset="yelp"                        # dataset used for training
model="GMF"                           # model type: GMF or NeuMF-end
seed=2025                             # random seed
gpu="0"                               # GPU ID
epoch_eval=10                         # epoch to start evaluation
top_k="50 100"                        # top-k metrics as a list of values
batch_size=1024                       # batch size
temp1=0.2                             # temp 1
temp2=0.5                             # temp 2
userfact1=0.5                         # user factor 1
userfact2=0.0                         # user factor 2

mkdir -p logs/${dataset};
log_path=logs/${dataset}/${model}_${seed}_${batch_size}.log;
echo "log_path=${log_path}";

# Run the script with all parameters
python -u main.py \
    --dataset $dataset \
    --model $model \
    --seed $seed \
    --gpu $gpu \
    --epoch_eval $epoch_eval \
    --top_k $top_k \
    --batch_size $batch_size \
    --temp1 $temp1 \
    --temp2 $temp2 \
    --userfact1 $userfact1 \
    --userfact2 $userfact2 2>&1 | tee "$log_path";