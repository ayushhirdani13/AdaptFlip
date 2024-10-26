DIR="logs/$1"
mkdir -p $DIR
nohup python -u main_user_wise.py --dataset=${1:-"movielens"} --model=${2:-"NeuMF"} --drop_rate=${3:-"0.2"} --num_gradual=${4:-"30000"} --best_k=${5:-"3"} > $DIR/${2}_${3}-${4}@${5}.log 2>&1 &