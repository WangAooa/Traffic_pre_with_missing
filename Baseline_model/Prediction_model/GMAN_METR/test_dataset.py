import pandas as pd
import numpy as np

# df = pd.read_csv('/home/wangao/Traffic_prediction_with_missing_value/METR-LA/dataset/metr-la(time).csv', header=0, index_col=0)
# print(df.index[:5])
# df_index = pd.to_datetime(df.index)
# print(df_index[:5])
#
# Time = df.index
# dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
# # timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
# #             // Time.freq.delta.total_seconds()
# timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
#             // (60 * 5)
# timeofday = np.reshape(timeofday, newshape = (-1, 1))
# Time = np.concatenate((dayofweek, timeofday), axis = -1)
# print(Time.shape)
# print(Time[:10])

SE_file = "../GMAN_PEMS(M)/data/SE(PEMS).txt"
f = open(SE_file, mode = 'r')
lines = f.readlines()
temp = lines[0].split(' ')
N, dims = int(temp[0]), int(temp[1])
SE = np.zeros(shape = (N, dims), dtype = np.float32)
for line in lines[1 :]:
    temp = line.split(' ')
    print(temp)
    index = int(temp[0])
    SE[index] = temp[1 :]
    print(SE[index][-5:])
print(SE.shape)
print(SE[-5:,-5:])