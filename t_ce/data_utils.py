import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict

from torch.utils.data import Dataset

# Constants
FILE_SUFFIXES = {
    'train': '.train.rating',
    'valid': '.valid.rating',
    'test': '.test.positive',
    'test_all': '.test.rating'
}
COLUMN_NAMES = ["user", "item", "true_label"]
COLUMN_DTYPES = {"user": np.int32, "item": np.int32, "true_label": np.int32}
def load_data(dataset, datapath):
    # Check if datapath exists
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"Datapath '{datapath}' does not exist")

    # Load training data
    train_file = os.path.join(datapath, f"{dataset}{FILE_SUFFIXES['train']}")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file '{train_file}' does not exist")
    train_data = pd.read_csv(train_file, sep="\t", header=None, names=COLUMN_NAMES, dtype=COLUMN_DTYPES)
    train_data_list = train_data[["user", "item"]].values
    train_data_true_label = train_data["true_label"].values


    # Load validation data
    valid_file = os.path.join(datapath, f"{dataset}{FILE_SUFFIXES['valid']}")
    if not os.path.exists(valid_file):
        raise FileNotFoundError(f"Valid file '{valid_file}' does not exist")
    valid_data = pd.read_csv(valid_file, sep="\t", header=None, names=COLUMN_NAMES, dtype=COLUMN_DTYPES)
    valid_data_list = valid_data[["user", "item"]].values
    valid_data_true_label = valid_data["true_label"].values

    # Create user-item matrix
    user_num = train_data["user"].max() + 1
    item_num = train_data["item"].max() + 1
    rows = train_data_list[:, 0]
    cols = train_data_list[:, 1]
    train_mat = sp.csr_matrix((np.ones_like(rows), (rows, cols)), shape=(user_num, item_num)).todok()

    user_pos = defaultdict(list)
    for user, item in np.concatenate((train_data_list, valid_data_list)):
        user_pos[user].append(item)

    # Load test data
    test_file = os.path.join(datapath, f"{dataset}{FILE_SUFFIXES['test']}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file '{test_file}' does not exist")
    test_data_pos = defaultdict(list)
    with open(test_file, "r") as f:
        for line in f.readlines():
            user, item, _ = line.strip().split('\t')
            user, item = int(user), int(item)
            test_data_pos[user].append(item)

    test_file = os.path.join(datapath, f"{dataset}{FILE_SUFFIXES['test_all']}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file '{test_file}' does not exist")
    test_df = pd.read_csv(test_file, sep="\t", header=None, names=COLUMN_NAMES, dtype=COLUMN_DTYPES)

    return (
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
    )

class NCF_Dataset(Dataset):
    def __init__(self, user_num, item_num, features, train_mat, true_labels, is_training=0, num_ng=1):
        super(NCF_Dataset, self).__init__()
        self.features = features
        self.true_labels = true_labels
        self.train_mat = train_mat
        self.is_training = is_training
        self.num_ng = num_ng
        self.labels = np.ones(len(self.features), dtype=np.int32)

        self.user_num = user_num
        self.item_num = item_num

    def ng_sample(self):
        if self.num_ng == 0:
            self.features_fill = self.features
            self.labels_fill = self.labels
            self.true_labels_fill = self.true_labels
        else:
            assert self.is_training != 2, "Sampling only for training mode"
            self.negative_samples = []
            for _ in range(self.num_ng):
                for u, _ in self.features:
                    j = np.random.randint(self.item_num)
                    while (u,j) in self.train_mat:
                        j = np.random.randint(self.item_num)
                    self.negative_samples.append((u,j))
            
            self.negative_samples = np.array(self.negative_samples)
            self.features_fill = np.concatenate((self.features, self.negative_samples))
            self.labels_fill = np.concatenate((self.labels, np.zeros(self.negative_samples.shape[0], dtype=np.int32)))
            self.true_labels_fill = np.concatenate((self.true_labels, np.zeros(self.negative_samples.shape[0], dtype=np.int32)))
            assert self.features_fill.shape[0] == self.labels_fill.shape[0]
            assert self.features_fill.shape[0] == self.true_labels_fill.shape[0]

    def __len__(self):
        return len(self.features) * (self.num_ng + 1)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training != 2 else self.features_ps
        labels = self.labels_fill if self.is_training != 2 else self.labels
        true_labels = self.true_labels_fill if self.is_training != 2 else self.true_labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        true_label = true_labels[idx]

        return user, item, label, true_label
    
class NCF_UserWise_Dataset(Dataset):
    def __init__(self, user_num, item_num, features, train_mat, true_labels, is_training=0, num_ng=1):
        super(NCF_UserWise_Dataset, self).__init__()
        self.features = features
        self.true_labels = true_labels
        self.train_mat = train_mat
        self.is_training = is_training
        self.num_ng = num_ng
        self.labels = np.ones(len(self.features), dtype=np.int32)

        self.user_num = user_num
        self.item_num = item_num

    def ng_sample(self):
        if self.num_ng == 0:
            self.features_fill = self.features
            self.labels_fill = self.labels
            self.true_labels_fill = self.true_labels
        else:
            assert self.is_training != 2, "Sampling only for training mode"
            self.negative_samples = []
            for _ in range(self.num_ng):
                for u, _ in self.features:
                    j = np.random.randint(self.item_num)
                    while (u,j) in self.train_mat:
                        j = np.random.randint(self.item_num)
                    self.negative_samples.append((u,j))
            
            self.negative_samples = np.array(self.negative_samples)
            self.features_fill = np.concatenate((self.features, self.negative_samples))
            self.labels_fill = np.concatenate((self.labels, np.zeros(self.negative_samples.shape[0], dtype=np.int32)))
            self.true_labels_fill = np.concatenate((self.true_labels, np.zeros(self.negative_samples.shape[0], dtype=np.int32)))
            assert self.features_fill.shape[0] == self.labels_fill.shape[0]
            assert self.features_fill.shape[0] == self.true_labels_fill.shape[0]

    def __len__(self):
        return self.user_num

    def __getitem__(self, user_id):
        features = self.features_fill if self.is_training != 2 else self.features_ps
        labels = self.labels_fill if self.is_training != 2 else self.labels
        true_labels = self.true_labels_fill if self.is_training != 2 else self.true_labels

        user_mask = features[:, 0] == user_id
        user = features[user_mask, 0]
        item = features[user_mask, 1]
        labels = labels[user_mask]
        true_labels = true_labels[user_mask]

        return user, item, labels, true_labels
    
class NCF_ItemWise_Dataset(Dataset):
    def __init__(self, user_num, item_num, features, train_mat, true_labels, is_training=0, num_ng=1):
        super(NCF_ItemWise_Dataset, self).__init__()
        self.features = features
        self.true_labels = true_labels
        self.train_mat = train_mat
        self.is_training = is_training
        self.num_ng = num_ng
        self.labels = np.ones(len(self.features), dtype=np.int32)

        self.user_num = user_num
        self.item_num = item_num

    def ng_sample(self):
        if self.num_ng == 0:
            self.features_fill = self.features
            self.labels_fill = self.labels
            self.true_labels_fill = self.true_labels
        else:
            assert self.is_training != 2, "Sampling only for training mode"
            self.negative_samples = []
            for _ in range(self.num_ng):
                for u, _ in self.features:
                    j = np.random.randint(self.item_num)
                    while (u,j) in self.train_mat:
                        j = np.random.randint(self.item_num)
                    self.negative_samples.append((u,j))
            
            self.negative_samples = np.array(self.negative_samples)
            self.features_fill = np.concatenate((self.features, self.negative_samples))
            self.labels_fill = np.concatenate((self.labels, np.zeros(self.negative_samples.shape[0], dtype=np.int32)))
            self.true_labels_fill = np.concatenate((self.true_labels, np.zeros(self.negative_samples.shape[0], dtype=np.int32)))
            assert self.features_fill.shape[0] == self.labels_fill.shape[0]
            assert self.features_fill.shape[0] == self.true_labels_fill.shape[0]

    def __len__(self):
        return self.item_num

    def __getitem__(self, item_id):
        features = self.features_fill if self.is_training != 2 else self.features_ps
        labels = self.labels_fill if self.is_training != 2 else self.labels
        true_labels = self.true_labels_fill if self.is_training != 2 else self.true_labels

        item_mask = features[:, 0] == item_id
        user = features[item_mask, 0]
        item = features[item_mask, 1]
        labels = labels[item_mask]
        true_labels = true_labels[item_mask]

        return user, item, labels, true_labels
