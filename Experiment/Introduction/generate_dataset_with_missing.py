from __future__ import absolute_import

import argparse
import numpy as np
import pandas as pd
import sys
sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")
from utils import *
from pathlib import Path


#model_type = ["BTMF", "brits"]
model_type = ["BTMF_r10","BTMF", "brits"]
missing_rate = [2,4, 6, 8]

dataset_type = ["PEMS(M)_NR", "METR-LA_NR"]
origin_vector = ["/PEMS(M)/dataset/V_228.csv", "/METR-LA/dataset/metr.csv"]

root_path = "/home/wangao/Traffic_prediction_with_missing_value"



def main(mask_path, origin_vector, save_path):
    # df = pd.read_hdf(args.traffic_df_filename, header=None).values.astype(float)
    origin_data = pd.read_csv(origin_vector, header=None).values.astype(float)
    mask = np.load(mask_path)

    scaler = StandardScaler(origin_data.mean(), origin_data.std())
    origin_data = scaler.transform(origin_data)

    missing_data = origin_data * mask

    print(origin_data.shape)
    # print(origin_data[:5,:5])
    # print(im_data[:5,:5])
    #
    # print(origin_data[-5:-1,-5:-1])
    # print(im_data[-5:-1,-5:-1])

    length = origin_data.shape[0]

    train_len, test_len, val_len = (int)(length * 0.6), (int)(length * 0.2), (int)(length * 0.2)
    # print(train_len, test_len, val_len)
    missing_train, origin_train = missing_data[0 : train_len], origin_data[0 : train_len]
    missing_test, origin_test = missing_data[train_len : train_len + test_len], origin_data[train_len : train_len + test_len]
    missing_val, origin_val = missing_data[train_len + test_len : length], origin_data[train_len + test_len : length]

    # print(train.shape)
    # print(test.shape)


    # print(scaler.mean, scaler.std)


    generate_dataset(missing_train,origin_train,  12, 12, 'train_pretask', scaler, save_path)
    generate_dataset(missing_test,origin_test,  12, 12, 'test_pretask', scaler, save_path)
    generate_dataset(missing_val,origin_val,  12, 12, 'val_pretask', scaler, save_path)


def generate_dataset(im_data, origin_data, his_len , pre_len, type, scaler,save_path):
    length, n_route = im_data.shape[0], im_data.shape[1]

    mean_std = np.array([scaler.mean, scaler.std])
    # im_data = scaler.transform(im_data)
    # origin_data = scaler.transform(origin_data)


    batch_size = length - his_len - pre_len + 1
    x ,y_label= [], []

    for i in range(batch_size):
        x_t = im_data[i : i + his_len].reshape(his_len, n_route, 1)
        #x_t[mask_t == 0] = 0
        y_t = origin_data[i + his_len : i + his_len + pre_len].reshape(pre_len, n_route, 1)

        x.append(x_t)
        y_label.append(y_t)

    x = np.stack(x, 0)
    y = np.stack(y_label, 0)

    print(x.shape)
    # print(x_mask.shape)
    # print(y_missing.shape)
    print(y.shape)
    # print(time_lag_mx.shape)
    #Path('{}{}'.format(args.output_dir, args.missing_rate)).mkdir(parents=True, exist_ok=True)

    save_path = save_path  + '_{}.npz'.format(type)
    np.savez(save_path, x = x, y = y, mean_std = mean_std)




if __name__ == "__main__":
#     model_type = ["BTMF", "brits"]
#
#     missing_rate = [2,4, 6, 8]
#
#     dataset_type = ["PEMS(M)", "METR-LA"]
#     origin_vector = ["/PEMS(M)/dataset/V_228.csv", "/METR-LA/dataset/metr.csv"]
#
#     root_path = "/home/wangao/Traffic_prediction_with_missing_value"
#
#     #
    for miss_rate in [2,8]:
        mask_path = root_path + "/PEMS(M)_NR/dataset/NR/mask_{}.npy".format(miss_rate)
        origin_vector_path = root_path + "/PEMS(M)/dataset/V_228.csv"
        save_path = "./{}_{}".format(dataset_type[0],miss_rate)
        main(mask_path, origin_vector_path, save_path)