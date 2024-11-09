# Configurations
dataset="movielens"              # dataset used for training
corruption_ratio=0.2              # corruption ratio
drop_rate=0                       # initial drop rate
num_gradual=0                 # epochs for linear increase in drop_rate
exponent=1                        # exponent for drop rate adjustment
lr=0.001                          # learning rate
batch_size=32                      # batch size
epochs=30                       # number of epochs for training
eval_freq=2000                    # evaluation frequency
top_k="3 5 10 20"                 # top-k metrics as a list of values
best_k=10                          # best-k for saving model checkpoint
factor_num=200                     # number of hidden factors
out='True'                        # if save outputs
gpu="0"                           # GPU ID

mkdir -p logs/${dataset};
log_path=logs/${dataset}/CDAE_${drop_rate}_${num_gradual}@${best_k}.log;
echo "log_path=${log_path}";

# Run the script with all parameters
python -u cdae.py \
    --dataset $dataset \
    --corruption_ratio $corruption_ratio \
    --drop_rate $drop_rate \
    --num_gradual $num_gradual \
    --exponent $exponent \
    --lr $lr \
    --batch_size $batch_size \
    --epochs $epochs \
    --eval_freq $eval_freq \
    --top_k $top_k \
    --best_k $best_k \
    --factor_num $factor_num \
    --out $out \
    --gpu $gpu 2>&1 | tee "$log_path"