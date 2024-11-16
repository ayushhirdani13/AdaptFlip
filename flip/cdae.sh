# Configurations
dataset="ml-1m"              # dataset used for training
corruption_ratio=0.2              # corruption ratio
W=2                       # Window Size
alpha=1                 # alpha in Q3 + alpha * IQR
lr=0.001                          # learning rate
batch_size=64                      # batch size
epochs=10                       # number of epochs for training
eval_freq=2000                    # evaluation frequency
top_k="3 5 10 20"                 # top-k metrics as a list of values
best_k=10                          # best-k for saving model checkpoint
factor_num=256                     # number of hidden factors
out='True'                        # if save outputs
gpu="0"                           # GPU ID

mkdir -p logs/${dataset}/loss;
log_path=logs/${dataset}/loss/CDAE_${W}_${alpha}@${best_k}.log;
echo "log_path=${log_path}";

# Run the script with all parameters
python -u cdae.py \
    --dataset $dataset \
    --corruption_ratio $corruption_ratio \
    --W $W \
    --alpha $alpha \
    --lr $lr \
    --batch_size $batch_size \
    --epochs $epochs \
    --eval_freq $eval_freq \
    --top_k $top_k \
    --best_k $best_k \
    --factor_num $factor_num \
    --out $out \
    --gpu $gpu 2>&1 | tee "$log_path"