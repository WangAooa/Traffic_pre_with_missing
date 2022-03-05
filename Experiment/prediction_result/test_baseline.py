import os

root_path = "/home/wangao/Traffic_prediction_with_missing_value"
data_path = "/data/wangao"
model = "GMAN"

SE_file = [root_path + "/Baseline_model/Prediction_model/GMAN_PEMS(M)/data/SE(PEMS).txt"]


missing_rate = [8]
#missing_rate = [2]

# dataset_type = ["PEMS(M)_NR", "METR-LA_NR"]
dataset_type = ["PEMS(M)_NR"]

save_type = ["PEMS(M)_NR"]

im_model_type = ["N"]
# im_model_type = ["brits"]

absolute_train_path = root_path + "/Baseline_model/Prediction_model/GMAN_METR/train.py"

origin_vector = [data_path + "/PEMS(M)/dataset/V_228.csv"]
# SE_file = [root_path + "/Baseline_model/Prediction_model/GMAN_nav-beijing/data/SE(nav-beijing).txt"
horrizon = 12

for miss_rate in missing_rate:
    for im_model in im_model_type:
        for i in range(len(dataset_type)):
            traffic_file = data_path + "/{}/dataset/NR/mask_{}.npy".format(dataset_type[i],miss_rate)
            model_file = data_path + "/result/missing_rate_{}/{}/{}/{}/model".format(miss_rate,save_type[i],model,im_model)
            log_file = data_path + "/result/missing_rate_{}/{}/{}/{}/log".format(miss_rate,save_type[i],model,im_model)
            save_dir = data_path + "/result/missing_rate_{}/{}/{}/{}".format(miss_rate,save_type[i],model,im_model)

            command = "python {}  --Q={} --traffic_file=\"{}\" --full_traffic_file=\"{}\" --SE_file=\"{}\" --model_file=\"{}\" --log_file=\"{}\" --save_dir=\"{}\""\
                .format(absolute_train_path, horrizon,traffic_file,origin_vector[i],SE_file[i], model_file,log_file, save_dir)

            os.system(command)

#还差7 birts

# num_node = [228]
# missing_rate = [8]
# # missing_rate = [3,5,7]
# adj_path = ["/PEMS(M)/dataset/W_228_normalized.npy"]
#
# dataset_type = ["PEMS(M)_NR"]
# # dataset_type = ["PEMS(M)_NR_3", "METR-LA_NR_3"]
#
# # im_model_type = ["BTMF_r10"]
# im_model_type = ["BTMF(50)"]
# horrizon = 12
#
# absolute_train_path = root_path + "/Baseline_script/ASTGCN_test.py"
#
# #Graph_Wavenet
# for miss_rate in missing_rate:
#     for i in range(len(num_node)):
#         for im_model in im_model_type:
#             command = "python {}  --num_node={} --horrizon={} --missing_rate={} --adj_path=\"{}\" --dataset_type=\"{}\" --im_model_type=\"{}\"" \
#                 .format(absolute_train_path,num_node[i],horrizon,miss_rate, adj_path[i],dataset_type[i],im_model)
#             os.system(command)