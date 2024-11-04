import os
from time import time
import random
import argparse

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import models
import data_utils
import evaluate
from loss import truncated_loss_cdae

def parse_args():
    datasets = os.listdir("../data")
    parser = argparse.ArgumentParser(description="Run CDAE, Normal and T-CE")
    parser.add_argument("--dataset",
        type=str,
        help=f"dataset used for training, options: {datasets}, default: movielens",
        default="movielens",
        choices=datasets)
    parser.add_argument("--corruption_ratio",
        type=float,
        default=0.2,
        help="corruption ratio, default: 0.2")
    parser.add_argument("--drop_rate",
        type=float,
        help="drop rate, default: 0.2",
        default=0.2)
    parser.add_argument("--num_gradual",
        type=int,
        default=30000,
        help="how many epochs to linearly increase drop_rate, default: 30000",)
    parser.add_argument("--exponent",
        type=float,
        default=1,
        help="exponent of the drop rate {0.5, 1, 2}, default: 1",)
    parser.add_argument("--lr",
        type=float,
        default=0.001,
        help="learning rate, default: 0.001",)
    parser.add_argument("--batch_size",
        type=int,
        default=32,
        help="batch size for training , default: 32",)
    parser.add_argument("--epochs",
        type=int,
        default=30,
        help="training epoches, default: 30")
    parser.add_argument("--eval_freq",
        type=int,
        default=2000,
        help="the freq of eval, default: 2000")
    parser.add_argument("--top_k",
        type=int,
        nargs='+',
        default=[3, 5, 10, 20],
        help="compute metrics@top_k, default: [3, 5, 10, 20]")
    parser.add_argument("--best_k",
        type=int,
        default=3,
        help="Best K for saving model, i.e., recall@best_k, default: 3")
    parser.add_argument("--factor_num",
        type=int,
        default=200,
        help="predictive factors numbers in the model, default: 200",)
    parser.add_argument("--out",
        type=str,
        default=True,
        help="save model or not, default: True")
    parser.add_argument("--gpu",
        type=str,
        default="0",
        help="gpu card ID, default: 0")
    
    args = parser.parse_args()
    if args.out in ["False", "false", "0"]:
        args.out = False
    else:
        args.out = True

    assert args.best_k in args.top_k, "best_k should be in top_k"
    args.best_k_ind = args.top_k.index(args.best_k)

    return args

def worker_init_fn(worker_id):
    np.random.seed(2024 + worker_id)


def drop_rate_schedule(iteration):
    drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
    if iteration < args.num_gradual:
        return drop_rate[iteration]
    else:
        return args.drop_rate
    
def get_results_dict(results, top_k):
    results_dict = {}
    for i, k in enumerate(top_k):
        results_dict[f"Recall@{k}"] = results["recall"][i]
        results_dict[f"NDCG@{k}"] = results["NDCG"][i]
        results_dict[f"Precision@{k}"] = results["precision"][i]
        results_dict[f"MRR@{k}"] = results["MRR"][i]
    return results_dict

########################### Test #####################################
@torch.no_grad()
def test(model, test_data_pos, user_pos, observed_mat):
    top_k = args.top_k
    model.eval()
    precision, recall, NDCG, MRR = evaluate.test_all_users_cdae(
        model, item_num, test_data_pos, user_pos, top_k, observed_mat, device
    )

    test_results = {
        "precision": precision,
        "recall": recall,
        "NDCG": NDCG,
        "MRR": MRR,
    }

    test_results_dict = get_results_dict(test_results, top_k)

    print(f"################### TEST ######################")
    print(pd.DataFrame(test_results, index=[f"@{i}" for i in args.top_k]).round(4).head())
    print("################### TEST END ######################\n")

    return recall[args.best_k_ind], test_results_dict

########################### Eval #####################################
@torch.no_grad()
def evalModel(model, valid_loader, count):
    model.eval()
    eval_loss = 0.0
    for i, (user, item_mat, _) in enumerate(valid_loader):
        user = user.cuda()
        item_mat = item_mat.float().cuda()

        prediction = model(user, item_mat)
        loss = truncated_loss_cdae(prediction, item_mat, drop_rate_schedule(count))
        eval_loss += loss.item()
    return eval_loss
    
if __name__ == '__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(2024) # cpu
    torch.cuda.manual_seed(2024) #gpu
    np.random.seed(2024) #numpy
    random.seed(2024) #random and transforms
    torch.backends.cudnn.deterministic=True # cudnn
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATASET = args.dataset
    DATAPATH = f"../data/{DATASET}"
    MODEL_DIR = f"models/{DATASET}"
    RESULT_DIR = f"results/{DATASET}"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    MODEL_FILE = f"CDAE_{args.drop_rate}-{args.num_gradual}@{args.best_k}.pth"
    args.model_path = os.path.join(MODEL_DIR, MODEL_FILE)

    print("Configurations:")
    for k, v in vars(args).items():
        print(k, ":", v)

    (
        user_num,
        item_num,
        train_mat,
        train_data_true,
        valid_mat,
        valid_data_true,
        observed_mat,
        user_pos,
        test_data_pos
    ) = data_utils.load_data_cdae(DATASET, DATAPATH)

    print("Data Loaded.")
    print(f"Users: {user_num}, Items: {item_num}")
    print(f"Number of Interactions: {train_mat.nnz}")
    print(f"Test Users: {len(test_data_pos)}")

    # Prepare Dataset
    train_dataset = data_utils.CDAE_Data(train_mat=train_mat, user_num=user_num, item_num=item_num, true_label=train_data_true)
    valid_dataset = data_utils.CDAE_Data(train_mat=valid_mat, user_num=user_num, item_num=item_num, true_label=valid_data_true)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

    model = models.CDAE(user_num=user_num, item_num=item_num, factor_num=args.factor_num, corruption_ratio=args.corruption_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    ########################## Training  ##########################
    count = 0
    best_loss, best_recall = 1e9, 0.0
    best_recall_idx = 0
    test_results = []
    start_time = time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for i, (user, item_mat, true_label) in enumerate(train_loader):
            user = user.cuda()
            item_mat = item_mat.float().cuda()

            prediction = model(user, item_mat)
            loss = truncated_loss_cdae(prediction, item_mat, drop_rate_schedule(count))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch[{epoch+1:03d}/{args.epochs:03d}], Train Loss: {epoch_loss:.4f}")

        eval_loss = evalModel(model, valid_loader, count)
        print(f"Epoch[{epoch+1:03d}/{args.epochs:03d}], Eval Loss: {eval_loss:.4f}")
        curr_recall, curr_test_results = test(model, test_data_pos, user_pos, observed_mat)
        curr_test_results["Validation Loss"] = eval_loss
        if curr_recall > best_recall:
            best_recall = curr_recall
            best_recall_idx = epoch
            if args.out == True:
                torch.save(
                    model,
                    args.model_path
                )
        test_results.append(curr_test_results)

    end_time = time()
    print("############################## Training End. ##############################")
    print("Training Time: {:.2f} seconds".format(end_time - start_time))
    print(f"Best Recall@{args.best_k}: {best_recall:.4f} at Epoch {best_recall_idx+1}")
    best_results = test_results[best_recall_idx]
    print(f"Best Test Results: ")
    best_results_df = pd.DataFrame(
    {
        "precision": [best_results[f"Precision@{k}"] for k in args.top_k],
        "recall": [best_results[f"Recall@{k}"] for k in args.top_k],
        "NDCG": [best_results[f"NDCG@{k}"] for k in args.top_k],
        "MRR": [best_results[f"MRR@{k}"] for k in args.top_k],
    },index=[f"@{k}" for k in args.top_k]
    ).round(4)

    print(best_results_df)

    results_df = pd.DataFrame(test_results).round(4)
    if args.out == True:
        results_df.to_csv(os.path.join(RESULT_DIR, f"{args.model}_{args.drop_rate}_{args.num_gradual}@{args.best_k}.csv"), index=False, float_format="%.4f")