from __future__ import absolute_import

import argparse
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from utils import *
from pathlib import Path


# missing_rate = [2,4, 6, 8]
missing_rate = [3,5,7]

dataset_type = ["PEMS(M)_NR", "METR-LA_NR"]
origin_vector = ["/PEMS(M)/dataset/V_228.csv", "/METR-LA/dataset/metr.csv"]

# root_path = "/home/wangao/Traffic_prediction_with_missing_value"
root_path = ".."

continue_mask_length = 4


def main(miss_rate,origin_vector, save_path,mask_save_path):
    # df = pd.read_hdf(args.traffic_df_filename, header=None).values.astype(float)
    origin_data = pd.read_csv(origin_vector, header=None).values.astype(float)
    #print(origin_data.shape)

    T, N = origin_data.shape[0], origin_data.shape[1]
    T = T - (T % continue_mask_length)
    origin_data = origin_data[:T, :].transpose()

    # scaler = StandardScaler(origin_data.mean(), origin_data.std())
    # origin_data = scaler.transform(origin_data)

    print(origin_data.shape)
    origin_data = origin_data.reshape((N, int(T/4), continue_mask_length))

    mask = np.ones((N * int(T/continue_mask_length), continue_mask_length))
    miss_index = np.random.choice(mask.shape[0], int(mask.shape[0] * miss_rate * 0.1), False)
    mask[miss_index,:] = 0
    mask = mask.reshape(origin_data.shape)

    # RN_data = origin_data * mask
    # RN_data = scaler.inverse_transform(RN_data).reshape(N,T).transpose()  #T*N
    mask = mask.reshape(N,T).transpose()

    # np.savetxt(save_path, my_matrix, delimiter=',')
    np.save(mask_save_path,mask)
    print(mask[:7,:7])





    # print(origin_data[:5,:5])
    # print(im_data[:5,:5])
    #
    # print(origin_data[-5:-1,-5:-1])
    # print(im_data[-5:-1,-5:-1])

    # length = min(im_data.shape[0],origin_data.shape[0])
    #
    # train_len, test_len, val_len = (int)(length * 0.6), (int)(length * 0.2), (int)(length * 0.2)
    # # print(train_len, test_len, val_len)
    # im_train, origin_train = im_data[0 : train_len], origin_data[0 : train_len]
    # im_test, origin_test = im_data[train_len : train_len + test_len], origin_data[train_len : train_len + test_len]
    # im_val, origin_val = im_data[train_len + test_len : length], origin_data[train_len + test_len : length]
    #
    # # print(train.shape)
    # # print(test.shape)
    #
    # scaler = StandardScaler(im_data.mean(), im_data.std())
    # # print(scaler.mean, scaler.std)
    #
    #
    # generate_dataset(im_train,origin_train,  12, 12, 'train_pretask', scaler, save_path)
    # generate_dataset(im_test,origin_test,  12, 12, 'test_pretask', scaler, save_path)
    # generate_dataset(im_val,origin_val,  12, 12, 'val_pretask', scaler, save_path)


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


    for miss_rate in missing_rate:
        for i in range(len(dataset_type)):
            origin_vector_path = root_path + origin_vector[i]

            Path(root_path + "/{}/dataset/NR".format(dataset_type[i])).mkdir(parents=True, exist_ok=True)

            save_path = root_path + "/{}/dataset/NR/V_{}.csv".format(dataset_type[i], miss_rate)
            mask_save_path = root_path + "/{}/dataset/NR/mask_{}.npy".format(dataset_type[i], miss_rate)
            main(miss_rate, origin_vector_path, save_path, mask_save_path)