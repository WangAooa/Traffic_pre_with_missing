import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

MAE = np.array([
    [[2.95, 2.98, 3.07, 4.62],
    [2.82, 2.85, 2.98, 3.90],
     [2.70, 2.76, 2.83, 3.10],
     [2.78, 2.79, 2.86, 3.94]],

    [[7.04, 7.26, 7.51, 11.27],
     [6.89, 6.84, 7.31, 10.04],
     [6.51, 6.67, 6.89, 7.6],
     [6.63, 6.61, 7.03, 9.72]],

    [[5.65, 5.71, 5.80, 8.33],
     [5.47, 5.56, 5.80, 7.54],
     [5.36, 5.46, 5.58, 6.05],
     [5.55, 5.58, 5.64, 7.48]]
])

MAPE = np.array([
    [[7.04, 7.26, 7.51, 11.27],
     [6.89, 6.84, 7.31, 10.04],
     [6.51, 6.67, 6.89, 7.6],
     [6.63, 6.61, 7.03, 9.72]],

    [[6.97, 7.13, 7.47, 8.21],
     [6.62, 6.71, 7.04, 7.71],
     [6.51, 6.67, 6.89, 7.6],
     [6.52, 6.61, 6.96, 7.59]],

    [[6.61, 6.8, 7.08, 7.88],
    [6.61, 6.7, 6.96, 7.88],
    [6.51, 6.67, 6.89, 7.6],
    [6.65, 6.71, 6.95, 7.71]]
])


# type = ['Dimention of ST-Block (r\'$D$\')', 'Layer of ST-Block (r\'$L$\')', 'Middle Dimension of FCs (r\'K$\')']
type = [r'$D$', r'$L$', r'$K$']
model_type = ['missing rate=20%', 'missing rate=40%','missing rate=60%','missing rate=80%']
color_type = ['b', 'g','y','m','r','c']
x = np.array([['16','32','64','128'],
              ['1', '2', '3', '4'],
              ['64', '128', '256','512']])

x_ln_dim = ['64', '128', '256','512']
x_ln_MAPE = np.array([
    [6.66, 6.78, 7.06, 7.63],
      [6.54, 6.74, 6.96, 7.68],
      [6.61, 6.8, 7.08, 7.88],
      [6.61, 6.7, 6.96, 7.88],
      [6.51, 6.67, 6.89, 7.6],
      [6.65, 6.71, 6.95, 7.71]])

# fig, axes = plt.subplots(1,3,figsize=(12,4))
# fig, axes = plt.subplots(1,3,figsize=(15,4))
# for i in range(2):
#     for j in range(len(model_type)):
#         print((i,j))
#         # print(MAE[i,:,j])
#         print(MAPE[i,:,j])
#         axes[i].plot(x[i],MAPE[i,:,j],label=model_type[j],color=color_type[j],marker='^',linestyle='--')
#
#     axes[i].set_xlabel(type[i])
#     axes[i].set_ylabel('MAPE')
#     axes[i].grid(linestyle='--')
#
# for j in range(len(model_type)):
#     axes[2].plot(x_ln_dim,x_ln_MAPE[:,j],label=model_type[j],color=color_type[j],marker='^',linestyle='--')
#
# axes[2].set_xlabel(type[2])
# axes[2].set_ylabel('MAPE')
# axes[2].grid(linestyle='--')
#
# # axes[0].legend(bbox_to_anchor=(0.93,0.7))
# # plt.legend(loc=(0,0) ,ncol = 5)
# plt.legend(loc='center', bbox_to_anchor=(-0.7,1.1),ncol=5)
# plt.savefig('./hyperparemters.pdf')
# plt.show()

size = 14
# plt.figure(figsize=(25,4.7))
# # plt.xticks( size=size)
# # plt.yticks( size=size)
# for i in range(3):
#     plt.subplot(1,3,i+1,)
#     for j in range(len(model_type)):
#         plt.plot(x[i],MAPE[i,:,j],label=model_type[j],color=color_type[j],marker='^',linestyle='--')
#
#     plt.xticks(size=size)
#     plt.yticks(size=size)
#     plt.xlabel(type[i], labelpad=0.1,fontdict={'size': size})
#     plt.ylabel("MAPE(%)", labelpad=0.1, fontdict={'size': size})
#     plt.grid(linestyle='--')
# plt.legend(loc='center', bbox_to_anchor=(-0.7,1.1),ncol=5,prop={'size':size})
# plt.savefig('./hyperparemters.pdf')
# plt.show()

fig, axes = plt.subplots(1,3,figsize=(18,4))
for i in range(3):
    for j in range(len(model_type)):
        axes[i].plot(x[i],MAPE[i,:,j],label=model_type[j],color=color_type[j],marker='^',linestyle='--')
        axes[i].tick_params(axis="x", labelsize=size)

        axes[i].tick_params(axis="y", labelsize=size)
        axes[i].set_xlabel(type[i], labelpad=0.1, fontdict={'size': size})
        axes[i].set_ylabel("MAPE(%)", labelpad=0.1, fontdict={'size': size})
        axes[i].grid(linestyle='--')
plt.legend(loc='center', bbox_to_anchor=(-0.7,1.1),ncol=5,prop={'size':size})
plt.savefig('./hyperparemters.pdf')
plt.show()