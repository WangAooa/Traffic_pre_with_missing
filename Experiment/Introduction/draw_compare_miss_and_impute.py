import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import torch
sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")

root_path = "/home/wangao/Traffic_prediction_with_missing_value"
NR_data = "./mask_8.npy"
full_data = "../../PEMS(M)/dataset/V_228.csv"

impute_data = "./V_brits_8.npy"

origin_data = pd.read_csv(full_data, header=None).values.astype(float)
impute_data = np.load(impute_data)
mask = np.load(NR_data)

print('origin_data shape :{}'.format(origin_data.shape))
print('impute_data shape :{}'.format(impute_data.shape))
# mae,mape,mse = MAE(torch.from_numpy(impute_data[:-1]),0,torch.from_numpy(origin_data))
mae,mape,mse = MAE(torch.from_numpy(impute_data),0,torch.from_numpy(origin_data))
print((mae,mape,mse))

# i = 10
# for i in range(12672 // 288):
#     print(i)
#     origin_data_list = origin_data[i*288: (i+1) * 288,0]
#     impute_data_list = impute_data[i*288:(i+1)*288,0]
#     mask_list = mask[:288,0]
#
#     plt.plot(origin_data_list,label='origin')
#     plt.plot(impute_data_list,label='impute')
#     plt.legend()
#     plt.show()
mask_list = mask[2880:288*11,0]
origin_data_list = origin_data[2880:288*11,0]
impute_data_list = impute_data[2880:288*11,0]

plt.plot([i for i in range(288)], origin_data_list)
plt.plot([i for i in range(288)], impute_data_list)

plt.xlim(0,288)
plt.ylim(0,90)

type = ['b--', 'b-']
index = 0
while index < 288:
    temp_index = index + 1
    while (temp_index < 288) and (mask_list[temp_index] == mask_list[index]):
        temp_index = temp_index + 1
    if int(mask_list[index]) == 0:
        plt.fill_between([i for i in range(index,temp_index)],0, 90,color='#D3D3D3')
    index = temp_index


#plt.show()
plt.savefig('./compare_miss_and_impute.png')

