import argparse
import os
import sys

import pandas as pd
import numpy as np
from torch.utils.data import *
import time
import csv
import torch
from pathlib import Path

sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")

from Baseline_model.Imputation_model.BTMF import *

torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--rnn_dim", type=int, default=64)
    parser.add_argument("--num_node", type=int, default=228)
    parser.add_argument("--num_rnn_layer", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--horrizon", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--missing_rate", type=int, default=8)
    parser.add_argument("--adj_path", type=str, default="/PEMS(M)/dataset/W_228_normalized.npy")
    parser.add_argument("--vector_datapath", type=str, default="/nav-beijing/dataset/bj_V_28000to40000.csv")
    parser.add_argument("--dataset_type", type=str, default="nav-beijing")
    parser.add_argument("--model_type", type=str, default='BTMF')
    parser.add_argument("--root_path", type=str, default="/data/wangao")
    parser.add_argument("--rank", type=int, default=50)

    args = parser.parse_args()
    print(args)

    Path(args.root_path +"/result/missing_rate_{}/{}/{}".format(args.missing_rate,args.dataset_type,args.model_type)).mkdir(parents=True, exist_ok=True)

    file = args.root_path +"/result/missing_rate_{}/{}/{}/test_result.txt".format(args.missing_rate,args.dataset_type,args.model_type)

    df = pd.read_csv(args.root_path + args.vector_datapath, header=None).values.astype(float)
    #df = np.ones((100,10))
    #NR
    # mask_path = args.root_path + "/{}/dataset/NR/mask_{}.npy".format(args.dataset_type, args.missing_rate)
    # mask = np.load(mask_path)
    #
    # num_time,num_node = df.shape
    # dense_mat = np.transpose(df)
    # sparse_mat = np.transpose(df * mask)

    #random missing
    num_time,num_node = df.shape
    dense_mat = np.transpose(df)

    mask = np.ones(num_node * num_time)
    miss_index = np.random.choice(mask.shape[0], int(mask.shape[0] * args.missing_rate * 0.1), False)
    mask[miss_index] = 0
    mask = mask.reshape(dense_mat.shape)
    sparse_mat = dense_mat * mask

    rank = args.rank
    time_lags = np.array([1, 2, 288])
    init = {"W": 0.1 * np.random.randn(num_node, rank), "X": 0.1 * np.random.randn(num_time, rank)}
    burn_iter = 200
    gibbs_iter = 200
    #mat_hat, W, X, A = BTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter, file)
    mat_hat, W, X, A = BTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter,file)

    mat_hat = np.transpose(mat_hat)
    # print("matrix :")
    # print(dense_mat[:5,:5])
    # print("imputed matrix ")
    # print(mat_hat[:5,:5])
    np.save(args.root_path +"/{}/dataset/V_{}_{}.npy".format(args.dataset_type, args.model_type, args.missing_rate), mat_hat)
    # np.save("./V_{}_{}.npy".format(args.dataset_type, args.model_type, args.missing_rate), mat_hat)


    # train_data_path = '{}/save/{}/train.npz'.format(args.dataset_type,args.missing_rate)
    # test_data_path = '{}/save/{}/test.npz'.format(args.dataset_type,args.missing_rate)
    #
    # adj = np.load(args.adj_path)
    # adj = torch.from_numpy(adj).to(device).to(torch.float)
    # print(adj[:5, :5])

if __name__ == '__main__':
    main()
    #test()