import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset


class board_data(Dataset):
    def __init__(self, dataset):  # dataset = np.array of (s, p, v)
        self.X = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.int64(self.X[idx]), self.y_p[idx], self.y_v[idx]


def load_state(net, iteration):
    base_path = "./model_data/"
    checkpoint_path = os.path.join(base_path,
                                   "AlphaZero_iter%d.pt" % (iteration))
    start_epoch, checkpoint = 0, None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'])
    return start_epoch


def load_results(iteration):
    """ Loads saved results if exists """
    losses_path = "./model_data/losses_per_epoch_iter%d.pkl" % iteration
    if os.path.isfile(losses_path):
        losses_per_epoch = load_data(
            "./model_data/losses_per_epoch_iter%d.pkl" % iteration)
    else:
        losses_per_epoch = []
    return losses_per_epoch


def save_data(data, file):
    fw = open(file, 'wb')
    pickle.dump(data, fw)
    fw.close()


def load_data(file):
    fr = open(file, 'rb')
    data = pickle.load(fr)
    fr.close()
    return data
