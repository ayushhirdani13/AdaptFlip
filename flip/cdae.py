import os
from time import time
import random
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import models
import data_utils
import evaluate

def parse_args():
    datasets = os.listdir("../data")
    parser = argparse.ArgumentParser(description="Run CDAE, Flip")
    parser.add_argument("--dataset",
        type=str,
        help=f"dataset used for training, options: {datasets}, default: movielens",
        default="movielens",
        choices=datasets)
    parser.add_argument("--corruption_ratio",
        type=float,
        default=0.2,
        help="corruption ratio, default: 0.2")
    parser.add_argument("--W",
        type=int,
        help="Window size, default: 2",
        default=2)
    parser.add_argument("--alpha",
        type=float,
        default=1,
        help="alpha in Q3 + alpha * IQR, default: 1",)
    parser.add_argument("--batch_mode",
        type=str,
        default="random",
        help="batch mode, default: user",
        choices=["neighbor", "random"])
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

    # print(f"################### TEST ######################")
    # print(pd.DataFrame(test_results, index=[f"@{i}" for i in args.top_k]).round(4).head())
    # print("################### TEST END ######################\n")

    return recall[args.best_k_ind], test_results_dict

########################### Eval #####################################
@torch.no_grad()
def evalModel(model, valid_loader, epoch, valid_log, device='cpu'):
    model.eval()
    eval_loss = 0.0
    flip_inds_buffer = np.empty((0,2), dtype=int)
    for i, (user, item_mat, label_mat, _) in enumerate(valid_loader):
        user = user.to(device)
        item_mat = item_mat.float().to(device)

        prediction = model(user, item_mat)
        loss_all = F.binary_cross_entropy_with_logits(prediction, item_mat, reduction='none')
        loss = torch.mean(loss_all)
        eval_loss += loss.item()

        pos_mask = label_mat > 0
        pos_indices = torch.argwhere(pos_mask).cpu().numpy()
        user_cpu = user.cpu().numpy()
        for u_ind, it in pos_indices:
            u = user_cpu[u_ind]
            valid_log[(u, it)] += loss_all[u_ind, it].item()
        if (epoch + 1) % args.W == 0:
            flip_inds = flipper(user_cpu, pos_indices, valid_log, args.W)
            flip_inds_buffer = np.concatenate((flip_inds_buffer, flip_inds), axis=0)

    if (epoch + 1) % args.W == 0:
        print("Valid Dataset State")
        valid_loader.dataset.flip_labels(flip_inds_buffer)
        valid_log.clear()
        if args.out == True:
            valid_loader.dataset.save_state(epoch=epoch, mode='valid', SAVE_DIR=OUTPUT_SAVE_DIR)
    return eval_loss

def flipper(user, pos_indices, loss_log, W):
    avg_losses = np.array([(loss_log[(user[u], i)] / W) for u, i in pos_indices])
    Q1, Q3 = np.quantile(avg_losses, q=[0.25, 0.75])
    IQR = Q3 - Q1
    threshold = Q3 + args.alpha * IQR
    threshold_mask = avg_losses > threshold
    flip_inds = pos_indices[threshold_mask]
    flip_inds[:,0] = user[flip_inds[:,0]]
    return flip_inds
    
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
    MODEL_DIR = f"models/{DATASET}/loss"
    RESULT_DIR = f"results/{DATASET}/loss"
    OUTPUT_SAVE_DIR = f"outputs/{DATASET}/loss"
    MODEL_DIR += f'/{args.batch_mode}'
    RESULT_DIR += f'/{args.batch_mode}'
    OUTPUT_SAVE_DIR += f'/{args.batch_mode}'
    OUTPUT_SAVE_DIR += f'/CDAE_{args.W}_{args.alpha}@{args.best_k}'
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)

    MODEL_FILE = f"CDAE_{args.W}_{args.alpha}@{args.best_k}.pth"
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

    if args.batch_mode == "neighbor":
        train_dataset = data_utils.CDAE_Neighbor_Data(train_mat=train_mat, user_num=user_num, item_num=item_num, true_label=train_data_true, group_size=args.batch_size)
        valid_dataset = data_utils.CDAE_Neighbor_Data(train_mat=valid_mat, user_num=user_num, item_num=item_num, true_label=valid_data_true, group_size=args.batch_size)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    else:
        train_dataset = data_utils.CDAE_Data(train_mat=train_mat, user_num=user_num, item_num=item_num, true_label=train_data_true)
        valid_dataset = data_utils.CDAE_Data(train_mat=valid_mat, user_num=user_num, item_num=item_num, true_label=valid_data_true)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

    model = models.CDAE(user_num=user_num, item_num=item_num, factor_num=args.factor_num, corruption_ratio=args.corruption_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    ########################## Training  ##########################
    count = 0
    best_loss, best_recall = 1e9, 0.0
    best_recall_idx = 0
    test_results = []
    start_time = time()

    train_log = defaultdict(float)
    valid_log = defaultdict(float)
    flip_inds_buffer = np.empty((0,2), dtype=int)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        if (epoch+1) % args.W == 0:
            print("#################### Flipping Epoch ##########################")

        for i, (user, item_mat, label_mat, true_label) in enumerate(train_loader):
            user = user.to(device)
            item_mat = item_mat.float().to(device)

            prediction = model(user, item_mat)
            loss_all = F.binary_cross_entropy_with_logits(prediction, item_mat, reduction='none')
            loss = torch.mean(loss_all)

            with torch.no_grad():    
                pos_mask = label_mat > 0
                pos_indices = torch.argwhere(pos_mask).cpu().numpy()
                user_cpu = user.cpu().numpy()
                for u_ind, it in pos_indices:
                    u = user_cpu[u_ind]
                    train_log[(u, it)] += loss_all[u_ind, it].item()
                if (epoch + 1) % args.W == 0:
                    flip_inds = flipper(user_cpu, pos_indices, train_log, args.W)
                    flip_inds_buffer = np.concatenate((flip_inds_buffer, flip_inds), axis=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            count += 1
        
        if (epoch + 1) % args.W == 0:
            print("Train Dataset State")
            train_loader.dataset.flip_labels(flip_inds_buffer)
            flip_inds_buffer = np.empty((0,2), dtype=int)
            train_log.clear()
            valid_log.clear()
            if args.out == True:
                train_loader.dataset.save_state(epoch=epoch, mode='train', SAVE_DIR=OUTPUT_SAVE_DIR)

        epoch_loss = epoch_loss / len(train_loader)

        eval_loss = evalModel(model, valid_loader, epoch, valid_log, device)
        print(f"Epoch[{epoch+1:03d}/{args.epochs:03d}], Train Loss: {epoch_loss}, Eval Loss: {eval_loss}")
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
        results_df.to_csv(os.path.join(RESULT_DIR, f"CDAE_{args.W}_{args.alpha}@{args.best_k}.csv"), index=False, float_format="%.4f")