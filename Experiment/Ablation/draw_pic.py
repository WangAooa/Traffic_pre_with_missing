import argparse
import os

import numpy as np
import torch
from torch.utils.data import *
import time
import csv
from pathlib import Path
import sys
from matplotlib import pyplot as plt

sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")

MAE = [
    [[3.21, 3.28, 3.49, 4.13],
    [7.15, 4.84, 5.85, 5.58],
    [8.64, 8.64, 8.77, 8.64],
    [2.82, 2.84, 2.96, 3.18],
    [2.75, 2.79, 2.87, 3.10]],

    [[7.81, 8.09, 8.62, 10.59],
    [14.70, 11.51, 14.79, 15.39],
    [28.03, 27.98, 27.51, 28.04],
    [6.83, 6.92, 7.22, 7.82],
    [6.62, 6.71, 7.04, 7.71]],

    [[6.45, 6.50, 6.88, 7.97],
    [10.97, 7.97, 9.27, 9.21],
    [14.58, 14.55, 14.06, 14.53],
    [5.54, 5.56, 5.82, 6.24],
    [5.44, 5.49, 5.63, 6.07]]
]
MAPE = [
    [7.81, 8.09, 8.62, 10.59],
    [14.70, 11.51, 14.79, 15.39],
    [28.03, 27.98, 27.51, 28.04],
    [6.76, 6.85, 7.16, 7.75],
    [6.62, 6.71, 7.04, 7.71]
]
RMSE = [
    [6.45, 6.50, 6.88, 7.97],
    [10.97, 7.97, 9.27, 9.21],
    [14.58, 14.55, 14.06, 14.53],
    [5.54, 5.56, 5.82, 6.24],
    [5.44, 5.49, 5.63, 6.07]
]
type = ['MAE', 'MAPE(%)', 'RMSE']
model_type = ['No-adp', 'No-GCN','No-GRU','No-Res','GSTAE']
color_type = ['b', 'g','y','m','r']
x = ['20','40','60','80']

size = 14

# fig, axes = plt.subplots(1,3,figsize=(12,4))
# fig, axes = plt.subplots(1,3,figsize=(15,4))
# plt.xticks( size=size)
# plt.yticks( size=size)
# for i in range(3):
#     for j in range(len(model_type)):
#         axes[i].plot(x,MAE[i][j],label=model_type[j],color=color_type[j],marker='^',linestyle='--')
#
#
#     axes[i].set_xlabel('Missing rate(%)', fontdict={'size': size})
#     axes[i].set_ylabel(type[i], fontdict={'size': size})
#     axes[i].grid(linestyle='--')
#     # plt.xticks(fontproperties='Times New Roman', size=14)
#     # plt.yticks(fontproperties='Times New Roman', size=14)
#     # axes[i].set_xlabel('Missing rate(%)',fontdict={'family' : 'Times New Roman','size':14})
#     # axes[i].set_ylabel(type[i],fontdict={'family' : 'Times New Roman','size':14})
#     # axes[i].grid(linestyle='--')
# # axes[0].legend(bbox_to_anchor=(0.93,0.7))
# # plt.legend(loc=(0,0) ,ncol = 5)
# # plt.legend(loc='center', bbox_to_anchor=(-0.7,1.1),ncol=5,prop={'family' : 'Times New Roman','size':14})
# plt.legend(loc='center', bbox_to_anchor=(-0.7,1.1),ncol=5,prop={'size':size})
# # plt.savefig('./ablation.pdf')
# plt.show()

plt.figure(figsize=(18,4))
# plt.xticks( size=size)
# plt.yticks( size=size)
for i in range(3):
    plt.subplot(1,3,i+1)
    for j in range(len(model_type)):
        plt.plot(x,MAE[i][j],label=model_type[j],color=color_type[j],marker='^',linestyle='--')

    plt.xticks(size=size)
    plt.yticks( size=size)
    plt.xlabel('Missing rate(%)', labelpad=0.1, fontdict={'size': size})
    plt.ylabel(type[i], labelpad=0.1,fontdict={'size': size})
    plt.grid(linestyle='--')
    # plt.xticks(fontproperties='Times New Roman', size=14)
    # plt.yticks(fontproperties='Times New Roman', size=14)
    # axes[i].set_xlabel('Missing rate(%)',fontdict={'family' : 'Times New Roman','size':14})
    # axes[i].set_ylabel(type[i],fontdict={'family' : 'Times New Roman','size':14})
    # axes[i].grid(linestyle='--')
# axes[0].legend(bbox_to_anchor=(0.93,0.7))
# plt.legend(loc=(0,0) ,ncol = 5)
# plt.legend(loc='center', bbox_to_anchor=(-0.7,1.1),ncol=5,prop={'family' : 'Times New Roman','size':14})
plt.legend(loc='center', bbox_to_anchor=(-0.7,1.1),ncol=5,prop={'size':size})
# plt.savefig('./ablation.pdf')
plt.show()