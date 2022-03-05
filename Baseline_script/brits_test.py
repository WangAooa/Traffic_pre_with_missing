import argparse
import os

import numpy as np
import torch
from torch.utils.data import *
import time
import csv
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")

from utils import *
# from model.rits import Model_1 as Model
from BRITS_model.brits import brits as Model

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")

torch.set_num_threads(1)

#用训练号的模型生成补全的数据

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--rnn_dim", type=int, default=64)
    parser.add_argument("--num_node", type=int, default=1362)                                    #
    parser.add_argument("--num_rnn_layer", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--horrizon", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--missing_rate", type=int, default=8)                                  #
    parser.add_argument("--vector_datapath", type=str, default="/nav-beijing/dataset/bj_V_28000to40000.csv")  #
    parser.add_argument("--dataset_type", type=str, default="nav-beijing")                          #
    parser.add_argument("--model_type", type=str, default='brits')
    parser.add_argument("--root_path", type=str, default="/data/wangao")
    parser.add_argument("--continue_mask_length", type=int, default=4)

    args = parser.parse_args()
    print(args)

    # data process
    data = pd.read_csv(args.root_path + args.vector_datapath, header=None).values.astype(float)
    scaler = StandardScaler(data.mean(), data.std())

    length, n_route = data.shape[0], data.shape[1]

    length = length - (length % args.continue_mask_length)
    data = data[:length]

    data = scaler.transform(data)

    #NR (Non-Random missing)
    # mask = np.ones((n_route * int(length / args.continue_mask_length), args.continue_mask_length))
    # miss_index = np.random.choice(mask.shape[0], int(mask.shape[0] * args.missing_rate * 0.1), False)
    # mask[miss_index, :] = 0
    # mask = mask.reshape(n_route, length).transpose() #T*N
    # data = data * mask

    # RM (Random missing)
    mask = np.ones(length * n_route)
    miss_index = np.random.choice(mask.shape[0], int(mask.shape[0] * args.missing_rate * 0.1), False)
    mask[miss_index] = 0
    mask = mask.reshape(length, n_route)
    data[mask == 0] = 0

    batch_size = length // args.seq_len
    x, x_mask, time_lag_mx, time_lag_mx_reverse = [], [], [], []
    for i in range(batch_size):
        mask_t = mask[i * args.seq_len: (i+1) * args.seq_len].reshape(args.seq_len, n_route, 1)
        x_t = data[i * args.seq_len: (i+1) * args.seq_len].reshape(args.seq_len, n_route, 1)
        time_lag_mx_t = generate_time_lag(mask[i * args.seq_len: (i+1) * args.seq_len]).reshape(args.seq_len, n_route, 1)
        time_lag_mx_reverse_t = generate_time_lag(np.flip(mask[i * args.seq_len: (i+1) * args.seq_len], axis=0)).reshape(args.seq_len, n_route, 1)

        x.append(x_t)
        x_mask.append(mask_t)
        time_lag_mx.append(time_lag_mx_t)
        time_lag_mx_reverse.append(time_lag_mx_reverse_t)

    x = torch.from_numpy(np.stack(x, 0))
    x_mask = torch.from_numpy(np.stack(x_mask, 0))
    time_lag_mx = torch.from_numpy(np.stack(time_lag_mx, 0))
    time_lag_mx_reverse = torch.from_numpy(np.stack(time_lag_mx_reverse, 0))

    print(x.shape)

    #model
    model_impute = Model(args.input_dim, args.rnn_dim, args.batch, args.num_node, args.seq_len, args.horrizon)
    model_impute.load_state_dict(
        torch.load(args.root_path +'/result/missing_rate_{}/{}/{}/model.pt'.format(args.missing_rate,args.dataset_type,args.model_type)))
    model_impute = model_impute.to(device)
    model_impute.eval()

    impute_result = []

    step = x.shape[0] // args.batch
    split_num = args.seq_len * args.batch

    for i in range(step):
        x_temp = x[i * args.batch : (i+1) * args.batch].float().to(device)
        mask_temp = x_mask[i * args.batch : (i+1) * args.batch].float().to(device)
        time_lag_temp = time_lag_mx[i * args.batch : (i+1) * args.batch].float().to(device)
        time_lag_mx_reverse_temp = time_lag_mx_reverse[i * args.batch : (i+1) * args.batch].float().to(device)

        impute_temp,_ = model_impute(x_temp,mask_temp,time_lag_temp,time_lag_mx_reverse_temp)
        impute_result.append(torch.reshape(impute_temp, (split_num,n_route)).detach().cpu().numpy())

    impute_result = np.concatenate(impute_result,0)
    impute_result = scaler.inverse_transform(impute_result)
    #print(impute_result[:5,:5])

    # Path('/home/wangao/Traffic_prediction_with_missing_value'
    #         +"/{}/dataset/V_{}_{}.npy".format(args.dataset_type, args.model_type, args.missing_rate)).mkdir(parents=True, exist_ok=True)
    np.save(args.root_path + "/{}/dataset/V_{}_{}.npy".format(args.dataset_type, args.model_type, args.missing_rate), impute_result)

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--rnn_dim", type=int, default=64)
    parser.add_argument("--num_node", type=int, default=1362)  #
    parser.add_argument("--num_rnn_layer", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--horrizon", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--missing_rate", type=int, default=8)  #
    parser.add_argument("--vector_datapath", type=str, default="/nav-beijing/dataset/bj_V_28000to40000.csv")  #
    parser.add_argument("--dataset_type", type=str, default="nav-beijing")  #
    parser.add_argument("--model_type", type=str, default='brits')
    parser.add_argument("--root_path", type=str, default="/home/wangao/Traffic_prediction_with_missing_value")

    args = parser.parse_args()
    x = [i for i in range(556)]
    data = np.load(args.root_path + "/PEMS(M)/dataset/V_BTMF_2.npy")
    # y_1 = data[:556,0]
    print(data.shape)
    # data = np.load("../PEMS(M)/dataset/V_brits_2.npy")
    # y_2 = data[:556,0]
    # data = pd.read_csv("../PEMS(M)/dataset/V_228.csv", header=None).values.astype(float)
    # print(data.shape)
    # y_3 = data[:556,0]
    # data = np.load("../V_PEMS(M)_BTMF.npy")
    # print(data.shape)
    # y_4 = data[:556, 0]
    #
    # plt.plot(x,y_1,label='BTMF')
    # plt.plot(x,y_2,label='brits')
    # plt.plot(x,y_3, label='real')
    # plt.plot(x, y_4, label='BTMF_40')
    #
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()
    #test()