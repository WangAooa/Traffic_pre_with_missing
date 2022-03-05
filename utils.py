import numpy as np
import torch
import pandas as pd
from torch.utils.data import *

# class Data(Dataset):
#     def __init__(self, x, x_mask, y_missing, y, time_lag):
#         self.x = torch.from_numpy(x)
#         self.x_mask = torch.from_numpy(x_mask)
#         self.y_missing = torch.from_numpy(y_missing)
#         self.y = torch.from_numpy(y)
#         self.time_lag = time_lag
#
#         self.len = x.shape[0]
#
#     def __getitem__(self, item):
#         return self.x[item], self.x_mask[item], self.y_missing[item], self.y[item], self.time_lag[item]
#
#     def __len__(self):
#         return self.len
class Data(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

        self.len = x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.len

class Data_pastdata(Dataset):
    def __init__(self, x_lastweek, x_lastday, x, x_mask, y_missing, y, time_lag):
        self.x_lastweek = torch.from_numpy(x_lastweek)
        self.x_lastday = torch.from_numpy(x_lastday)
        self.x = torch.from_numpy(x)
        self.x_mask = torch.from_numpy(x_mask)
        self.y_missing = torch.from_numpy(y_missing)
        self.y = torch.from_numpy(y)
        self.time_lag = time_lag

        self.len = x.shape[0]

    def __getitem__(self, item):
        return self.x_lastweek[item], self.x_lastday[item], self.x[item], self.x_mask[item], self.y_missing[item], \
               self.y[item], self.time_lag[item]

    def __len__(self):
        return self.len

class Databidir(Dataset):
    def __init__(self, x, x_mask, y_missing, y, time_lag, time_lag_reverse):
        self.x = torch.from_numpy(x)
        self.x_mask = torch.from_numpy(x_mask)
        self.y_missing = torch.from_numpy(y_missing)
        self.y = torch.from_numpy(y)
        self.time_lag = time_lag
        self.time_lag_reverse = time_lag_reverse

        self.len = x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.x_mask[item], self.y_missing[item], self.y[item], self.time_lag[item], self.time_lag_reverse[item]

    def __len__(self):
        return self.len

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# def MAE(x, x_mask, y):
#     y[y < 0.01] = 0
#     #x_mask = 0 -> not missing;   x_mask = 1 -> missing
#     x_mask = -(x_mask - 1)
#
#     # x_mask[y[x_mask == 1] == 0] = 0
#     index_x_mask = (x_mask == 1).float()
#     index_y = (y == 0).float()
#     index = index_x_mask * index_y
#     x_mask[index == 1] = 0
#     # y_mask = (y != 0).float()
#     # y_mask /= y_mask.mean()
#     #
#     # loss = torch.abs(x - y)
#     # loss = loss * y_mask
#
#     impute_mae = torch.sum(torch.abs(x - y) * x_mask) / (torch.sum(x_mask) + 1e-5)
#     impute_mape = torch.abs(x - y) / y
#     impute_mape[impute_mape != impute_mape] = 0
#     impute_mape[torch.isinf(impute_mape)] = 0
#
#     impute_mape = torch.sum(impute_mape * x_mask) / (torch.sum(x_mask) + 1e-5)
#
#     return  impute_mae, impute_mape
def impute_MAE(x, x_mask, y):
    y[y < 0.01] = 0
    #x_mask = 0 -> not missing;   x_mask = 1 -> missing
    x_mask = -(x_mask - 1)

    # index_x_mask = (x_mask == 1).float()
    # index_y = (y == 0).float()
    # index = index_x_mask * index_y
    # x_mask[index == 1] = 0
    y_mask = (y != 0).float()
    x_mask = x_mask * y_mask
    
    # loss = torch.abs(x - y)
    # loss = loss * y_mask

    impute_mae = torch.sum(torch.abs(x - y) * x_mask) / (torch.sum(x_mask) + 1e-5)
    impute_mae[impute_mae != impute_mae] = 0

    impute_mape = torch.abs(x - y) / y
    impute_mape[impute_mape != impute_mape] = 0
    impute_mape[torch.isinf(impute_mape)] = 0

    impute_mape = torch.sum(impute_mape * x_mask) / (torch.sum(x_mask) + 1e-5)

    impute_mse = (x - y)**2
    impute_mse[impute_mse != impute_mse] = 0
    impute_mse = torch.sum(impute_mse * x_mask) / (torch.sum(x_mask) + 1e-5)

    return impute_mae, impute_mape,impute_mse

def MAE(x, x_mask, y):
    y[y < 0.01] = 0

    mask = (y != 0).float()
    mask /= mask.mean()
    mae = torch.abs(y - x)
    mae = mae * mask
    mae[mae != mae] = 0

    mape = mae / y
    mape[mape != mape] = 0
    mape[torch.isinf(mape)] = 0

    mse = mae ** 2
    mse[mse != mse] = 0
    return mae.mean(), mape.mean(), mse.mean()

    # mask = (y != 0).float()
    # mask /= np.mean(mask)
    #
    # mae = np.abs(y - x).astype(np.float32)
    # mse = np.square(mae)
    # mape = np.divide(mae, y)
    # mae = np.nan_to_num(mae * mask)
    # mae = np.mean(mae)
    # mse = np.nan_to_num(mse * mask)
    # mse = np.mean(mse)
    # mape = np.nan_to_num(mape * mask)
    # mape = np.mean(mape)

def masked_mae_loss(y_pred, y_true):
    y_true[y_true < 0.01] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0

    mape = loss / y_true
    mape[mape != mape] = 0
    mape[torch.isinf(mape)] = 0

    mse = loss ** 2
    mse[mse != mse] = 0

    return loss.mean(), mape.mean(), mse.mean()


def transfer_to_device(x, device):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :returns x shape (seq_len, batch_size, num_sensor, input_dim)
             y shape (horizon, batch_size, num_sensor, input_dim)
    """
    x = x.float()
    return x.to(device)

def thresholded_Gaussian_kernel(adj_path, normalized_k):
    adj = pd.read_csv(adj_path, header=None).values.astype(float)
    num_node = adj.shape[0]
    adj = adj[~np.isinf(adj[0])].flatten()
    std = adj.std()
    print(std)
    adj_mx = np.exp(-np.square(adj / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx.reshape((num_node,num_node))

def thresholded_Gaussian_kernel_STGCN(adj_path, sigma2 = 0.1, epsilon = 0.5):
    # get from STGCN https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/utils/math_graph.py
    adj = pd.read_csv(adj_path, header=None).values.astype(float)
    adj = adj / 10000
    print(adj[:5,:5])

    adj2 = adj * adj
    adj_mx = np.exp(-adj2 / sigma2)
    adj_mx[adj_mx < epsilon] = 0

    return adj_mx

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    #A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    print(diag[:10])
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def diffusion_adj(A):
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    print(D.shape)

    diag = np.reciprocal(D)
    A_wave = np.multiply(diag.reshape((-1, 1)), A)
    return A_wave

def generate_time_lag(mask):
    time_lag = np.ones(mask.shape)
    time_lag[0] = 0
    for i in range(1, time_lag.shape[0]):
        for j in range(time_lag.shape[1]):
            if mask[i - 1][j] == 0:
                time_lag[i][j] += time_lag[i - 1][j]
    return time_lag
if __name__ == '__main__':
    # adj = pd.read_csv('./PEMS(M)/dataset/W_228.csv', header=None).values.astype(float)
    # print(adj[:5,:5])
    adj = thresholded_Gaussian_kernel_STGCN('./PEMS(M)/dataset/W_228.csv')
    # adj = thresholded_Gaussian_kernel
    print(adj[:5, :5])
    test = True




    # x = np.load('./PEMS(M)/dataset/W_228.npy')
    # print(x[:5, :5])
    x = get_normalized_adj(adj)
    print(x[:5,:5])
    x = diffusion_adj(adj)
    print(x[:5,:5])
    # np.save('./PEMS(M)/dataset/W_228_normalized.npy', x)

    # x_mask[y[x_mask == 1] == 0] = 0

    print('beijing')
    adj = pd.read_csv('./nav-beijing/dataset/bj_A.csv', header=None).values.astype(float)
    print(adj[:5,:5])
    # # adj = thresholded_Gaussian_kernel('./PEMS(M)/dataset/W_228.csv', 0.5)
    #
    adj = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))
    print(adj[:5, :5])
    #
    # nor = get_normalized_adj(adj)
    # print(nor[:5,:5])
    #
    adj = diffusion_adj(adj)
    print(adj[:5,:5])
    np.save('./nav-beijing/dataset/W_1362_diffusion.npy', adj)
    print('beining')
    adj = np.load('./nav-beijing/dataset/W_1362.npy')
    print(adj[:5,:5])
    adj = np.load('./nav-beijing/dataset/W_1362_transpose_diffusion.npy')
    print(adj[:5,:5])