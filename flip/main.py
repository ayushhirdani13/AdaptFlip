import os
from time import time
import argparse
import numpy as np
import pandas as pd
import random
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import data_utils
import models
import evaluate

def parse_args():
    datasets = os.listdir("../data")
    parser = argparse.ArgumentParser(description="Run NCF, Flip based on Avg Loss")
    parser.add_argument("--dataset",
        type=str,
        help=f"dataset used for training, options: {datasets}, default: movielens",
        default="movielens",
        choices=datasets)
    parser.add_argument("--model",
        type=str,
        help="model used for training. options: GMF, NeuMF, default: NeuMF",
        default="NeuMF",
        choices=["GMF", "NeuMF"])
    parser.add_argument("--W",
        type=int,
        help="Window size, default: 2",
        default=2)
    parser.add_argument("--alpha",
        type=float,
        default=1,
        help="alpha in Q3 + alpha * IQR, default: 1",)
    parser.add_argument("--lr",
        type=float,
        default=0.001,
        help="learning rate, default: 0.001",)
    parser.add_argument("--dropout",
        type=float,
        default=0.0,
        help="dropout rate, default: 0.0",)
    parser.add_argument("--batch_by",
        type=str,
        default="none",
        help="batch by user, item or none, default: none",
        choices=["none", "user", "item"])
    parser.add_argument("--batch_mode",
        type=str,
        default="random",
        help="batch mode:neighbor or random, default: random",
        choices=["random", "neighbor"])
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

    return args

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
def evalModel(model, valid_loader, epoch, valid_log, valid_log_buffer, device='cuda'):
    model.eval()
    epoch_loss = 0
    valid_loader.dataset.ng_sample()
    flip_inds_buffer = set()
    for user, item, label, train_label, _, idx in valid_loader:
        user = user.to(device)
        item = item.to(device)
        train_label = train_label.float().to(device)
        # label = label.float().to(device)

        prediction = model(user, item)
        loss_all = F.binary_cross_entropy_with_logits(prediction, train_label, reduction='none')
        loss = torch.mean(loss_all)
        epoch_loss += loss.item()

        # Convert tensors to CPU and NumPy arrays for logging
        user_cpu = user.cpu().numpy()
        item_cpu = item.cpu().numpy()
        train_label_cpu = train_label.cpu().numpy()
        loss_all_cpu = loss_all.cpu().detach().numpy()
        idx_cpu = idx.cpu().numpy()
        label_cpu = label.cpu().numpy()
        pos_mask = label_cpu == 1

        pos_indices = np.where(pos_mask)[0]

        for ind in pos_indices:
            u = user_cpu[ind]
            i = item_cpu[ind]

            # Update valid_log in batch
            loss_val = loss_all_cpu[ind]
            valid_label = int(train_label_cpu[ind])
            valid_log_buffer.append([u, i, epoch, f"{loss_val:.4f}", valid_label])
            valid_log[(u, i)] += loss_val
        if (epoch+1) % args.W == 0:
            flip_inds = flipper(user_cpu[pos_mask], item_cpu[pos_mask], idx_cpu[pos_mask], valid_log, args.W)
            flip_inds_buffer.update(flip_inds)

    if (epoch+1) % args.W == 0:
        print("Valid Dataset State")
        valid_loader.dataset.flip_labels(list(flip_inds_buffer))
        flip_inds_buffer.clear()
        if args.out:
            valid_loader.dataset.save_state(epoch=epoch, mode='valid', SAVE_DIR=OUTPUT_SAVE_DIR)
    return epoch_loss

def flipper(user, item, idx, loss_log, W):
    user_item_pairs = np.column_stack((user, item))
    avg_losses = np.array([(loss_log[tuple(pair)] / W) for pair in user_item_pairs])

    # Calculate thresholds using IQR (Interquartile Range)
    avg_Q1, avg_Q3 = np.quantile(avg_losses, [0.25, 0.75])
    avg_IQR = avg_Q3 - avg_Q1
    avg_upper = avg_Q3 + args.alpha * avg_IQR

    # Determine indices where average loss exceeds the threshold
    flip_inds = idx[avg_losses > avg_upper].tolist()

    return flip_inds

def custom_collate_fn(batch):
    user_tensors = []
    item_tensors = []
    label_tensors = []
    train_label_tensors = []
    true_label_tensors = []
    idx_tensors = []

    for user, item, label, train_label, true_label, idx in batch:
        user_tensors.append(torch.tensor(user))
        item_tensors.append(torch.tensor(item))
        label_tensors.append(torch.tensor(label))
        train_label_tensors.append(torch.tensor(train_label))
        true_label_tensors.append(torch.tensor(true_label))
        idx_tensors.append(torch.tensor(idx))

    users = torch.cat(user_tensors, dim=0)
    items = torch.cat(item_tensors, dim=0)
    labels = torch.cat(label_tensors, dim=0)
    train_labels = torch.cat(train_label_tensors, dim=0)
    true_labels = torch.cat(true_label_tensors, dim=0)
    idxs = torch.cat(idx_tensors, dim=0)

    return users, items, labels, train_labels, true_labels, idxs

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
    MODEL_DIR = f"models/{DATASET}/loss"
    RESULT_DIR = f"results/{DATASET}/loss"
    OUTPUT_SAVE_DIR = f"outputs/{DATASET}/loss"
    RESULT_DIR += f'/{args.batch_mode}/{args.batch_by}'
    MODEL_DIR += f'/{args.batch_mode}/{args.batch_by}'
    OUTPUT_SAVE_DIR += f'/{args.batch_mode}/{args.batch_by}/{args.model}_{args.W}_{args.alpha}_{args.batch_size}@{args.best_k}'
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)

    MODEL_FILE = f"{args.model}_{args.W}_{args.alpha}_{args.batch_size}@{args.best_k}.pth"
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
    if args.batch_by == 'none':
        train_dataset = data_utils.NCF_Dataset(user_num, item_num, train_data_list, train_mat, train_data_true_label, is_training=1, num_ng=args.num_ng)
        valid_dataset = data_utils.NCF_Dataset(user_num, item_num, valid_data_list, train_mat, valid_data_true_label, is_training=0, num_ng=args.num_ng)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    else:
        if args.batch_mode == 'neighbor':
            train_dataset = data_utils.NCF_NeighborWise_Dataset(user_num, item_num, train_data_list, train_mat, train_data_true_label, is_training=1, num_ng=args.num_ng, group_size=args.batch_size, neighbor_type=args.batch_by)
            valid_dataset = data_utils.NCF_NeighborWise_Dataset(user_num, item_num, valid_data_list, train_mat, valid_data_true_label, is_training=0, num_ng=args.num_ng, group_size=args.batch_size, neighbor_type=args.batch_by)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)
        elif args.batch_by == 'user':
            train_dataset = data_utils.NCF_UserWise_Dataset(user_num, item_num, train_data_list, train_mat, train_data_true_label, is_training=1, num_ng=args.num_ng)
            valid_dataset = data_utils.NCF_UserWise_Dataset(user_num, item_num, valid_data_list, train_mat, valid_data_true_label, is_training=0, num_ng=args.num_ng)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)
        elif args.batch_by == 'item':
            train_dataset = data_utils.NCF_ItemWise_Dataset(user_num, item_num, train_data_list, train_mat, train_data_true_label, is_training=1, num_ng=args.num_ng)
            valid_dataset = data_utils.NCF_ItemWise_Dataset(user_num, item_num, valid_data_list, train_mat, valid_data_true_label, is_training=0, num_ng=args.num_ng)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)

    if args.model == "GMF":
        model = models.GMF(user_num, item_num, args.factor_num).to(device)
    elif args.model == "NeuMF":
        model = models.NeuMF(user_num, item_num, args.factor_num, args.mlp_layers, args.dropout).to(device)
    else:
        raise ValueError(f"No model named as {args.model}")
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ########################## Training  ##########################
    count = 0
    best_loss, best_recall = 1e9, 0.0
    best_recall_idx = 0
    test_results = []
    start_time = time()

    train_log = defaultdict(float)
    valid_log = defaultdict(float)

    train_log_buffer = []
    valid_log_buffer = []
    training_losses_buffer = []
    flip_inds_buffer = set()

    if args.out == True:
        RUNS_DIR = f"runs/{args.dataset}"
        os.makedirs(RUNS_DIR, exist_ok=True)
        training_losses_file = os.path.join(RUNS_DIR, f"training_losses_{args.model}_{args.W}_{args.alpha}_{args.batch_size}@{args.best_k}.csv")
        train_logs_file = os.path.join(RUNS_DIR, f"train_logs_{args.model}_{args.W}_{args.alpha}_{args.batch_size}@{args.best_k}.csv")
        valid_logs_file = os.path.join(RUNS_DIR, f"valid_logs_{args.model}_{args.W}_{args.alpha}_{args.batch_size}@{args.best_k}.csv")

        ## Clear File contents before run
        open(training_losses_file, 'w').close()
        open(train_logs_file, 'w').close()
        open(valid_logs_file, 'w').close()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        train_loader.dataset.ng_sample()

        if (epoch+1) % args.W == 0:
            print("#################### Flipping Epoch ##########################")
        for i, (user, item, label, train_label, true_label, idx) in enumerate(train_loader):
            user = user.to(device)
            item = item.to(device)
            train_label = train_label.float().to(device)
            # label = label.float().to(device)

            prediction = model(user, item)
            loss_all = F.binary_cross_entropy_with_logits(prediction, train_label, reduction='none')
            loss = torch.mean(loss_all)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # Convert tensors to CPU and NumPy arrays for logging
            user_cpu = user.cpu().numpy()
            item_cpu = item.cpu().numpy()
            train_label_cpu = train_label.cpu().int().numpy()
            loss_all_cpu = loss_all.cpu().detach().numpy()
            idx_cpu = idx.cpu().numpy()
            label_cpu = label.cpu().numpy()
            pos_mask = label_cpu == 1
            true_label_cpu = true_label.cpu().numpy()

            tp_loss = loss_all_cpu[(train_label_cpu == 1) & (true_label_cpu == 1)].sum()
            fp_loss = loss_all_cpu[(train_label_cpu == 0) & (true_label_cpu == 1)].sum()
            loss_val = loss.item()

            training_losses_buffer.append([count, f"{tp_loss:.4f}", f"{fp_loss:.4f}", f"{loss_val:.4f}"])

            pos_indices = np.where(pos_mask)[0]

            for ind in pos_indices:
                u = user_cpu[ind]
                i = item_cpu[ind]

                # Update train_log in batch
                loss_val = loss_all_cpu[ind]
                train_label = int(train_label_cpu[ind])
                train_log_buffer.append([u, i, epoch, f"{loss_val:.4f}", train_label])
                train_log[(u, i)] += loss_val
            if (epoch+1) % args.W == 0:
                flip_inds = flipper(user_cpu[pos_mask], item_cpu[pos_mask], idx_cpu[pos_mask], train_log, args.W)
                flip_inds_buffer.update(flip_inds)

            epoch_loss += loss.item()
            count += 1
        epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch[{epoch+1:03d}/{args.epochs:03d}], Train Loss: {epoch_loss:.4f}")

        if (epoch+1) % args.W == 0:
            print("Train Dataset State")
            train_loader.dataset.flip_labels(list(flip_inds_buffer))
            flip_inds_buffer.clear()
            train_log.clear()
            valid_log.clear()
            if args.out:
                train_loader.dataset.save_state(epoch=epoch, mode='train', SAVE_DIR=OUTPUT_SAVE_DIR)

        eval_loss = evalModel(model, valid_loader, epoch, valid_log, valid_log_buffer, device=device)
        print(f"Epoch[{epoch+1:03d}/{args.epochs:03d}], Eval Loss: {eval_loss:.4f}")
        curr_recall, curr_test_results = test(model, test_data_pos, user_pos)
        curr_test_results["Validation Loss"] = eval_loss


        if args.out == True:
            if len(training_losses_buffer) > 0:
                with open(training_losses_file, 'a', newline='') as f:
                    NAMES = ["epoch", "tp_loss", "fp_loss", "loss"]
                    df = pd.DataFrame(training_losses_buffer, columns=NAMES)
                    df.to_csv(f, mode='a',index=False, header=False)

            if len(train_log_buffer) > 0:
                with open(train_logs_file, 'a', newline='') as f:
                    NAMES = ["user", "item", "epoch", "loss", "train_label"]
                    df = pd.DataFrame(train_log_buffer, columns=NAMES)
                    df.to_csv(f, mode='a',index=False, header=False)

            if len(valid_log_buffer) > 0:
                with open(valid_logs_file, 'a', newline='') as f:
                    NAMES = ["user", "item", "epoch", "loss", "train_label"]
                    df = pd.DataFrame(valid_log_buffer, columns=NAMES)
                    df.to_csv(f, mode='a',index=False, header=False)

        training_losses_buffer.clear()
        valid_log_buffer.clear()
        train_log_buffer.clear()

        if curr_recall > best_recall:
            best_recall = curr_recall
            best_recall_idx = epoch
            if args.out:
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
        results_df.to_csv(os.path.join(RESULT_DIR, f"{args.model}_{args.W}_{args.alpha}_{args.batch_size}@{args.best_k}.csv"), index=False, float_format="%.4f")