# Configurations
dataset="movielens"              # dataset used for training
seed=2025                        # random seed for reproducibility
gpu="0"                         # GPU ID
epoch_eval=10                    # epoch to start evaluation
batch_size=1024                  # batch size
top_k="50 100"                   # top-k metrics as a list of values
temp1=0.2                        # temp1 parameter
temp2=0.5                        # temp2 parameter
userfact1=0.5                    # user factor 1
userfact2=0.0                    # user factor 2

mkdir -p logs/${dataset};
log_path=logs/${dataset}/CDAE_${dataset}_${temp1}_${temp2}_${userfact1}_${userfact2}.log;
echo "log_path=${log_path}";

# Run the script with all parameters
python -u main_CDAE.py \
    --dataset $dataset \
    --seed $seed \
    --gpu $gpu \
    --epoch_eval $epoch_eval \
    --batch_size $batch_size \
    --top_k $top_k \
    --temp1 $temp1 \
    --temp2 $temp2 \
    --userfact1 $userfact1 \
    --userfact2 $userfact2 2>&1 | tee "$log_path"
