import os

root_path = "/home/wangao/Traffic_prediction_with_missing_value"
data_path = "/data/wangao"
model = "GMAN"

SE_file = [root_path + "/Baseline_model/Prediction_model/GMAN_PEMS(M)/data/SE(PEMS).txt"
           ,root_path + "/Baseline_model/Prediction_model/GMAN_METR/data/SE(METR).txt"
           ,root_path + "/Baseline_model/Prediction_model/GMAN_PEMS(M)/data/SE(PEMS).txt"
           ,root_path + "/Baseline_model/Prediction_model/GMAN_METR/data/SE(METR).txt"]


missing_rate = [2,3,4,5,6,7,8]
#missing_rate = [2]

# dataset_type = ["PEMS(M)_NR", "METR-LA_NR"]
dataset_type = ["PEMS(M)", "METR-LA","PEMS(M)_NR", "METR-LA_NR"]

save_type = ["PEMS(M)_6", "METR-LA_6", "PEMS(M)_NR_6", "METR-LA_NR_6"]

im_model_type = ["brits", "BTMF","BTMF_r10"]
# im_model_type = ["brits"]

absolute_train_path = root_path + "/Baseline_model/Prediction_model/GMAN_METR/train.py"

origin_vector = [data_path + "/PEMS(M)/dataset/V_228.csv", data_path + "/METR-LA/dataset/metr.csv",
                 data_path + "/PEMS(M)/dataset/V_228.csv", data_path + "/METR-LA/dataset/metr.csv"]
# SE_file = [root_path + "/Baseline_model/Prediction_model/GMAN_nav-beijing/data/SE(nav-beijing).txt"]
#
#
# missing_rate = [8]
#
# dataset_type = ["nav-beijing"]
#
# save_type = ["nav-beijing"]
#
# im_model_type = ["brits", "BTMF"]
#
# absolute_train_path = root_path + "/Baseline_model/Prediction_model/GMAN_METR/train.py"
#
# origin_vector = [data_path + "/nav-beijing/dataset/bj_V_28000to40000.csv"]
horrizon = 6

for miss_rate in missing_rate:
    for im_model in im_model_type:
        for i in range(len(dataset_type)):
            traffic_file = data_path + "/{}/dataset/V_{}_{}.npy".format(dataset_type[i],im_model,miss_rate)
            model_file = data_path + "/result/missing_rate_{}/{}/{}/{}/model".format(miss_rate,save_type[i],model,im_model)
            log_file = data_path + "/result/missing_rate_{}/{}/{}/{}/log".format(miss_rate,save_type[i],model,im_model)
            save_dir = data_path + "/result/missing_rate_{}/{}/{}/{}".format(miss_rate,save_type[i],model,im_model)

            command = "python {}  --Q={} --traffic_file=\"{}\" --full_traffic_file=\"{}\" --SE_file=\"{}\" --model_file=\"{}\" --log_file=\"{}\" --save_dir=\"{}\""\
                .format(absolute_train_path, horrizon,traffic_file,origin_vector[i],SE_file[i], model_file,log_file, save_dir)

            os.system(command)

#还差7 birts