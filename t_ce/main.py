import os
from time import time
import argparse
import numpy as np
import pandas as pd
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import data_utils
import models
from loss import truncated_loss
import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Run T_CE & Normal NeuMF")
    parser.add_argument("--dataset",
        type=str,
        help="dataset used for training, options: movielens, default: movielens",
        default="movielens",)
    parser.add_argument("--model",
        type=str,
        help="model used for training. options: GMF, NeuMF, default: NeuMF",
        default="NeuMF",)
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
    parser.add_argument("--dropout",
        type=float,
        default=0.0,
        help="dropout rate, default: 0.0",)
    parser.add_argument("--batch_size",
        type=int,
        default=1024,
        help="batch size for training , default: 1024",)
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
        default=32,
        help="predictive factors numbers in the model, default: 32",)
    parser.add_argument("--mlp_layers",
        type=int,
        nargs='+',
        default=[256, 128, 64],
        help="number of layers in MLP model, default: [256, 128, 64]")
    parser.add_argument("--num_ng",
        type=int,
        default=1,
        help="sample negative items for training, default: 1")
    parser.add_argument("--out",
        default=True,
        help="save model or not, default: True")
    parser.add_argument("--gpu",
        type=str,
        default="0",
        help="gpu card ID, default: 0")

    return parser.parse_args()

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
def test(model, test_data_pos, user_pos):
    top_k = args.top_k
    model.eval()
    precision, recall, NDCG, MRR = evaluate.test_all_users(
        model, item_num, test_data_pos, user_pos, top_k, device
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
def evalModel(model, valid_loader, count, device='cuda'):
    model.eval()
    epoch_loss = 0
    valid_loader.dataset.ng_sample()
    for user, item, label, _ in valid_loader:
        user = user.to(device)
        item = item.to(device)
        label = label.float().to(device)

        prediction = model(user, item)
        loss = truncated_loss(prediction, label, drop_rate=drop_rate_schedule(count))
        epoch_loss += loss.item()
    return epoch_loss

def worker_init_fn(worker_id):
    np.random.seed(2024 + worker_id)

if __name__ == "__main__":
    args = parse_args()

    assert args.best_k in args.top_k, "best_k should be in top_k"
    args.best_k_ind = args.top_k.index(args.best_k)

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

    MODEL_FILE = f"{args.model}_{args.drop_rate}-{args.num_gradual}@{args.best_k}.pth"
    args.model_path = os.path.join(MODEL_DIR, MODEL_FILE)

    print("Configurations:")
    for k, v in vars(args).items():
        print(k, ":", v)

    (
        user_num,
        item_num,
        train_data_list,
        train_data_true_label,
        train_mat,
        valid_data_list,
        valid_data_true_label,
        user_pos,
        test_data_pos,
        test_df
    ) = data_utils.load_data(DATASET, DATAPATH)

    print("Data Loaded")
    print(f"Users: {user_num}, Items: {item_num}")
    print(f"Number of Interactions: {len(train_data_list)}")
    print(f"Test Users: {len(test_data_pos)}")

    ## Prepare Datasets
    train_dataset = data_utils.NCF_Dataset(user_num=user_num, item_num=item_num, features=train_data_list,train_mat=train_mat, true_labels=train_data_true_label, num_ng=args.num_ng)
    valid_dataset = data_utils.NCF_Dataset(user_num=user_num, item_num=item_num, features=valid_data_list,train_mat=train_mat, true_labels=valid_data_true_label, num_ng=args.num_ng)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    if args.model == "GMF":
        model = models.GMF(user_num, item_num, args.factor_num).to(device)
    elif args.model == "NeuMF":
        model = models.NeuMF(user_num, item_num, args.factor_num, args.mlp_layers, args.dropout).to(device)
    else:
        raise ValueError("No model named as {}".format(args.model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ########################## Training  ##########################
    count = 0
    best_loss, best_recall = 1e9, 0.0
    best_recall_idx = 0
    test_results = []
    losses = []
    start_time = time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        train_loader.dataset.ng_sample()
        for i, (user, item, label, true_label) in enumerate(train_loader):
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            prediction = model(user, item)
            loss = truncated_loss(prediction, label, drop_rate=drop_rate_schedule(count))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            losses.append(loss.item())
            count += 1
        epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch[{epoch+1:03d}/{args.epochs:03d}], Train Loss: {epoch_loss:.4f}")

        eval_loss = evalModel(model, valid_loader, count)
        print(f"Epoch[{epoch+1:03d}/{args.epochs:03d}], Eval Loss: {eval_loss:.4f}")
        curr_recall, curr_test_results = test(model, test_data_pos, user_pos)
        curr_test_results["Validation Loss"] = eval_loss

        if curr_recall > best_recall:
            best_recall = curr_recall
            best_recall_idx = epoch
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
    results_df.to_csv(os.path.join(RESULT_DIR, f"{args.model}_{args.drop_rate}_{args.num_gradual}@{args.best_k}.csv"), index=False, float_format="%.4f")