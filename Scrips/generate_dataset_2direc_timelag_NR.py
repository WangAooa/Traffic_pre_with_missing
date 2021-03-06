from __future__ import absolute_import

import argparse
import numpy as np
import pandas as pd
import sys
sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")
from utils import *
from pathlib import Path


def main(args):
    # df = pd.read_hdf(args.traffic_df_filename, header=None).values.astype(float)
    df = pd.read_csv(args.traffic_df_filename, header=None).values.astype(float)
    length = df.shape[0]

    train_len, test_len, val_len = (int)(length * 0.7), (int)(length * 0.2), (int)(length * 0.1)
    # print(train_len, test_len, val_len)
    train = df[0 : train_len]
    test = df[train_len : train_len + test_len]
    val = df[train_len + test_len : ]

    mask= np.load(args.mask_path)
    train_mask = mask[0 : train_len]
    test_mask = mask[train_len : train_len + test_len]
    val_mask = mask[train_len + test_len : length]

    scaler = StandardScaler(df.mean(), df.std())
    # print(scaler.mean, scaler.std)

    generate_dataset(train,  12, args.horrizon, args, 'train_bidir', scaler, train_mask)
    generate_dataset(test,  12, args.horrizon, args, 'test_bidir', scaler, test_mask)
    generate_dataset(val, 12, args.horrizon, args, 'val_bidir', scaler, val_mask)

def generate_time_lag(mask):
    time_lag = np.ones(mask.shape)
    time_lag[0] = 0
    for i in range(1, time_lag.shape[0]):
        for j in range(time_lag.shape[1]):
            if mask[i - 1][j] == 0:
                time_lag[i][j] += time_lag[i - 1][j]
    return time_lag

def generate_dataset(data, his_len  , pre_len, args, type, scaler,mask):
    length, n_route = data.shape[0], data.shape[1]

    mean_std = np.array([scaler.mean, scaler.std])
    data = scaler.transform(data)

    #generate the mask matrix, if mask[i] == 0, it means index i of data is missing;
    # if mask[i] == 1, it means the index i of data is not missing, and the total missing rate = args.missing_rate
    # mask = np.ones(length * n_route)
    # miss_index = np.random.choice(mask.shape[0], int(mask.shape[0] * args.missing_rate * 0.1), False)
    # mask[miss_index] = 0
    # mask = mask.reshape(length, n_route)

    #time_lag = generate_time_lag(mask)

    # based on the mask matrix, process the data???generate a new dataset whose miss_rate is args.missing_rate
    data_mask = data.copy()
    data_mask[mask == 0] = 0
    # print(scaler.transform(0))


    batch_size = length - his_len - pre_len + 1
    x, x_mask, y_missing, y_label, time_lag_mx, time_lag_mx_reverse= [], [], [], [], [],[]
    for i in range(batch_size):
        mask_t = mask[i : i + his_len].reshape(his_len, n_route, 1)
        y_missing_t = data[i : i + his_len].reshape(his_len, n_route, 1)
        x_t = data_mask[i : i + his_len].reshape(his_len, n_route, 1)
        #x_t[mask_t == 0] = 0
        y_t = data[i + his_len : i + his_len + pre_len].reshape(pre_len, n_route, 1)
        time_lag_mx_t = generate_time_lag(mask[i : i + his_len]).reshape(his_len, n_route, 1)
        time_lag_mx_reverse_t = generate_time_lag(np.flip(mask[i: i + his_len], axis=0)).reshape(his_len, n_route, 1)

        x.append(x_t)
        x_mask.append(mask_t)
        y_missing.append(y_missing_t)
        y_label.append(y_t)
        time_lag_mx.append(time_lag_mx_t)
        time_lag_mx_reverse.append(time_lag_mx_reverse_t)

    x = np.stack(x, 0)
    x_mask = np.stack(x_mask, 0)
    y_missing = np.stack(y_missing, 0)
    y = np.stack(y_label, 0)
    time_lag_mx = np.stack(time_lag_mx, 0)
    time_lag_mx_reverse = np.stack(time_lag_mx_reverse, 0)

    # print(x.shape)
    # print(x_mask.shape)
    # print(y_missing.shape)
    # print(y.shape)
    # print(time_lag_mx.shape)
    Path('{}{}'.format(args.output_dir, args.missing_rate)).mkdir(parents=True, exist_ok=True)

    save_path = '{}{}/{}.npz'.format(args.output_dir, args.missing_rate, type)
    np.savez(save_path, x = x,x_mask = x_mask, y_missing = y_missing, y = y, time_lag = time_lag_mx,
             time_lag_reverse = time_lag_mx_reverse, mean_std = mean_std)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="../PEMS(M)/save/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="../PEMS(M)/dataset/V_228.csv",
        help="Raw traffic readings.",
    )
    parser.add_argument(
        "--missing_rate", type=int, default=4
    )
    parser.add_argument(
        "--mask_path",
        type=str
    )
    parser.add_argument(
        "--horrizon",
        type=int
    )
    args = parser.parse_args()
    main(args)