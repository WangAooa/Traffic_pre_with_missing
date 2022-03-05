import os

num_node = [228,207]
# missing_rate = [2,4,6,8]
missing_rate = [2,3,4,5,6,7,8]
adj_path = ["/PEMS(M)/dataset/W_228_normalized.npy", "/METR-LA/dataset/W_207.npy"]

# dataset_type = ["PEMS(M)_6", "METR-LA_6"]
dataset_type = ["PEMS(M)_NR_6", "METR-LA_NR_6"]
horrizon = 6

#BTMF
for miss_rate in missing_rate:
    for i in range(len(num_node)):
        command = "python ./main_gwGRU.py  --num_node={} --horrizon={} --missing_rate={} --adj_path=\"{}\" --dataset_type=\"{}\"" \
            .format(num_node[i],horrizon,miss_rate, adj_path[i],dataset_type[i])
        os.system(command)


#PEMS，metr的_NR_3 没跑