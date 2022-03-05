import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

origin_data =  pd.read_csv('../../PEMS(M)/dataset/V_228.csv', header=None).values.astype(float)[288*7+11:,0]
print(origin_data.shape)
pre = np.load('./gwnet_GRU_rnn_layer_3_PEMS(M)_NR_8.npy')[288*7-12:]
print(pre.shape)

GMAN_pre = np.load('./GMAN_PEMS(M)_NR_8.npy',allow_pickle=True)[288*7-12:]
print(GMAN_pre.shape)

ASTGCN_pre = np.load('./ASTGCN_PEMS(M)_NR_8.npy') #8天开始
print(ASTGCN_pre.shape)

# 2 6 8
x  = [i for i in range(288)]
for day in range(100):
    print(day)
    plt.plot(origin_data[day*288 : (day+4)*288],label='origin_data')
    plt.plot(pre[day*288 : (day+4)*288], label='gwGRU_prediction')
    plt.plot(GMAN_pre[day * 288: (day + 4) * 288], label='GMAN_prediction')
    plt.plot(ASTGCN_pre[day * 288: (day + 4) * 288], label='ASTGCN_prediction')
    plt.legend()
    plt.show()

