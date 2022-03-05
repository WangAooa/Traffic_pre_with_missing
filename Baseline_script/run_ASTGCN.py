import os
num_node = [228,207,228,207]
missing_rate = [2,3,4,5,6,7,8]
# missing_rate = [3,5,7]
adj_path = ["/PEMS(M)/dataset/W_228_normalized.npy", "/METR-LA/dataset/W_207.npy",
            "/PEMS(M)/dataset/W_228_normalized.npy", "/METR-LA/dataset/W_207.npy"]

dataset_type = ["PEMS(M)_NR_3", "METR-LA_NR_3",
                "PEMS(M)_3", "METR-LA_3"]
# dataset_type = ["PEMS(M)_NR_3", "METR-LA_NR_3"]

# im_model_type = ["BTMF_r10"]
im_model_type = ["brits", "BTMF", "BTMF_r10"]
horrizon = 3

#Graph_Wavenet
for miss_rate in missing_rate:
    for i in range(len(num_node)):
        for im_model in im_model_type:
            command = "python ASTGCN_test.py  --num_node={} --horrizon={} --missing_rate={} --adj_path=\"{}\" --dataset_type=\"{}\" --im_model_type={}" \
                .format(num_node[i],horrizon,miss_rate, adj_path[i],dataset_type[i],im_model)
            os.system(command)


#完成： PEMS，METR， ramdom miss, borrizon =12

#未完成：NR， horrizon=3,6; Random 3 6