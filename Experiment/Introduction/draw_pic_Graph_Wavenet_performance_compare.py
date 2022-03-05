import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")
from utils import *

data_path_8 = "./8_pre.npy"
data_path_2 = "./2_pre.npy"
origin_path_8 = "./8_label.npy"

time_length = 228

pre_data_2 = np.load(data_path_2)
pre_data_8 = np.load(data_path_8)
origin_data = np.load(origin_path_8)


print(pre_data_2.shape)
print(pre_data_8.shape)
print(origin_data.shape)

pre_data_2 = pre_data_2[:time_length,5,0,0]
pre_data_8 = pre_data_8[:time_length,5,0,0]
origin_data = origin_data[:time_length,5,0,0]
x = [i for i in range(time_length)]
plt.plot(x, pre_data_2, label = 'missing_rate=2')
plt.plot(x, pre_data_8, label = 'missing_rate=8')
plt.plot(x, origin_data, label = 'label')
plt.xlabel("time step")
plt.ylabel("km/h")
# #plt.xticks(np.linspace(0,456,144))
plt.legend()
plt.show()
#plt.savefig("./compare_2_and_8.png")

