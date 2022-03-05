from __future__ import absolute_import

import argparse
import numpy as np
import pandas as pd
import sys
sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")
from utils import *
from pathlib import Path


#model_type = ["BTMF", "brits"]
# model_type = ["BTMF_r10","BTMF", "brits"]
# missing_rate = [2,3,4,5, 6,7, 8]
# # missing_rate = [3,5,7]
#
# dataset_type = ["PEMS(M)_NR", "METR-LA_NR"]
# # dataset_type = ["PEMS(M)", "METR-LA"]
#
# save_dataset_type = ["PEMS(M)_NR_6", "METR-LA_NR_6"]
# horrizon = 6
#
# origin_vector = ["/PEMS(M)/dataset/V_228.csv", "/METR-LA/dataset/metr.csv"]
#
# root_path = "/data/wangao"
model_type = ["brits", "BTMF"]
missing_rate = [8]
# missing_rate = [3,5,7]

dataset_type = ["nav-beijing"]

save_dataset_type = ["nav-beijing"]

origin_vector = ["/nav-beijing/dataset/bj_V_28000to40000.csv"]

root_path = "/data/wangao"
horrizon = 12

def main(im_vector, origin_vector, save_path):
    # df = pd.read_hdf(args.traffic_df_filename, header=None).values.astype(float)
    origin_data = pd.read_csv(origin_vector, header=None).values.astype(float)
    im_data = np.load(im_vector)

    print(origin_data.shape)
    print(im_data.shape)
    # print(origin_data[:5,:5])
    # print(im_data[:5,:5])
    #
    # print(origin_data[-5:-1,-5:-1])
    # print(im_data[-5:-1,-5:-1])

    length = min(im_data.shape[0],origin_data.shape[0])

    train_len, test_len, val_len = (int)(length * 0.6), (int)(length * 0.2), (int)(length * 0.2)
    # print(train_len, test_len, val_len)
    im_train, origin_train = im_data[0 : train_len], origin_data[0 : train_len]
    im_test, origin_test = im_data[train_len : train_len + test_len], origin_data[train_len : train_len + test_len]
    im_val, origin_val = im_data[train_len + test_len : length], origin_data[train_len + test_len : length]

    # print(train.shape)
    # print(test.shape)

    scaler = StandardScaler(im_data.mean(), im_data.std())
    # print(scaler.mean, scaler.std)


    generate_dataset(im_train,origin_train,  12, horrizon, 'train_pretask', scaler, save_path)
    generate_dataset(im_test,origin_test,  12, horrizon, 'test_pretask', scaler, save_path)
    generate_dataset(im_val,origin_val,  12, horrizon, 'val_pretask', scaler, save_path)


def generate_dataset(im_data, origin_data, his_len , pre_len, type, scaler,save_path):
    length, n_route = im_data.shape[0], im_data.shape[1]

    mean_std = np.array([scaler.mean, scaler.std])
    im_data = scaler.transform(im_data)
    origin_data = scaler.transform(origin_data)


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
#     for miss_rate in missing_rate:
#         for i in range(len(dataset_type)):
#             for im_model in model_type:
#                 impute_vector_path = root_path + "/{}/dataset/V_{}_{}.npy".format(dataset_type[i],im_model,miss_rate)
#                 origin_vector_path = root_path + origin_vector[i]
#                 save_path = root_path + "/{}/save/{}/{}".format(save_dataset_type[i],miss_rate,im_model)
#                 main(impute_vector_path, origin_vector_path, save_path)
    for miss_rate in missing_rate:
        for i in range(len(dataset_type)):
            for im_model in model_type:
                impute_vector_path = root_path + "/{}/dataset/V_{}_{}.npy".format(dataset_type[i],im_model,miss_rate)
                origin_vector_path = root_path + origin_vector[i]
                save_path = root_path + "/{}/save/{}/{}".format(save_dataset_type[i],miss_rate,im_model)
                Path(root_path + "/{}/save/{}".format(save_dataset_type[i],miss_rate)).mkdir(parents=True, exist_ok=True)
                main(impute_vector_path, origin_vector_path, save_path)