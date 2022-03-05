import sys
import numpy as np
from matplotlib import pyplot as plt

MAE = np.array([
    [[2.88, 2.95, 3.09, 3.33],
    [2.75, 2.79, 2.87, 3.10],
     [2.70, 2.76, 2.83, 3.10],
     [2.71, 2.74, 2.83, 3.09]],

    [[6.97, 7.13, 7.47, 8.21],
     [6.62, 6.71, 7.04, 7.71],
     [6.51, 6.67, 6.89, 7.6],
     [6.52, 6.61, 6.96, 7.59]],

    [[5.63, 5.69, 5.96, 6.34],
     [5.44, 5.49, 5.63, 6.07],
     [5.36, 5.46, 5.58, 6.05],
     [5.43, 5.45, 5.59, 6.10]]
])


type = ['MAE', 'MAPE(%)', 'RMSE']
model_type = ['missing rate=0.2', 'missing rate=0.4','missing rate=0.6','missing rate=0.8']
color_type = ['b', 'g','y','m','r']
x = ['1','2','3','4']


# fig, axes = plt.subplots(1,3,figsize=(12,4))
fig, axes = plt.subplots(1,3,figsize=(15,4))
for i in range(3):
    for j in range(len(model_type)):
        axes[i].plot(x,MAE[i,:,j],label=model_type[j],color=color_type[j],marker='^',linestyle='--')

    axes[i].set_xlabel('Rnn layers')
    axes[i].set_ylabel(type[i])
    axes[i].grid(linestyle='--')
# axes[0].legend(bbox_to_anchor=(0.93,0.7))
# plt.legend(loc=(0,0) ,ncol = 5)
plt.legend(loc='center', bbox_to_anchor=(-0.7,1.1),ncol=5)
plt.savefig('./rnn_layer_hyperparemters.pdf')
# plt.show()