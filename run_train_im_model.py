import os

num_node = [228, 207]
# missing_rate = [3,5,7]
missing_rate = [4,5,6,7,8]

adj_path = ["./PEMS(M)/dataset/W_228_normalized.npy", "./METR-LA/dataset/W_207.npy"]
# dataset_type = ["PEMS(M)_NR", "METR-LA_NR"]
dataset_type = ["PEMS(M)_3", "METR-LA_3"]
horrizon = 3

#train and val
for miss_rate in missing_rate:
    for i in range(len(dataset_type)):
        command = "python ./main_bidir.py --num_node={} --horrizon={} --missing_rate={} --adj_path=\"{}\" --dataset_type=\"{}\""\
            .format(num_node[i],horrizon,miss_rate, adj_path[i], dataset_type[i])
        os.system(command)

#test
for miss_rate in missing_rate:
    for i in range(len(dataset_type)):
        command = "python ./model_test_bidir.py --num_node={} --horrizon={} --missing_rate={} --adj_path=\"{}\" --dataset_type=\"{}\""\
            .format(num_node[i],horrizon,miss_rate, adj_path[i], dataset_type[i])
        os.system(command)