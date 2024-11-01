# Configurations
dataset="movielens"              # dataset used for training
model="NeuMF"                       # model type, e.g., GMF or NeuMF
W=2                       # Window Size
alpha=1                 # alpha in Q3 + alpha * IQR
lr=0.001                          # learning rate
dropout=0.0                       # dropout rate
batch_by='user'               # batch by : {'none','user', 'item'}
batch_mode='random'           # batch mode: {'random', 'neighbor'}
batch_size=1                      # batch size
epochs=30                       # number of epochs for training
eval_freq=2000                    # evaluation frequency
top_k="3 5 10 20"                 # top-k metrics as a list of values
best_k=3                          # best-k for saving model checkpoint
factor_num=32                     # number of latent factors
mlp_layers="128 64"                # MLP layer sizes
num_ng=1                          # negative samples for training
out='True'                        # if save outputs
gpu="0"                           # GPU ID

mkdir -p logs/${dataset}/grad/${batch_mode}/${batch_by};
log_path=logs/${dataset}/grad/${batch_mode}/${batch_by}/${model}_${W}-${alpha}@${best_k}.log;
echo "log_path=${log_path}";

# Run the script with all parameters
python -u main.py \
    --dataset $dataset \
    --model $model \
    --W $W \
    --alpha $alpha \
    --lr $lr \
    --dropout $dropout \
    --batch_by $batch_by \
    --batch_mode $batch_mode \
    --batch_size $batch_size \
    --epochs $epochs \
    --eval_freq $eval_freq \
    --top_k $top_k \
    --best_k $best_k \
    --factor_num $factor_num \
    --mlp_layers $mlp_layers \
    --num_ng $num_ng \
    --out $out \
    --gpu $gpu 2>&1 | tee "$log_path";