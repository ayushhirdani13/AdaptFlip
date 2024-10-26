DIR="logs/$1/user_wise"
mkdir -p $DIR
nohup python -u main_user_wise.py --dataset=${1:-"movielens"} --model=${2:-"NeuMF"} --drop_rate=${3:-"0.2"} --num_gradual=${4:-"30000"} --batch_size=${5:-"1"} --best_k=${6:-"10"} > $DIR/${2}_${3}-${4}_${5}@${6}.log 2>&1 &