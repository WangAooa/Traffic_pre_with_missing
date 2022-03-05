import os

# output_dir = ["/home/wangao/Traffic_prediction_with_missing_value/PEMS(M)/save/",
#               "/home/wangao/Traffic_prediction_with_missing_value/METR-LA/save/"]
#
# traffic_df_filename = ["/home/wangao/Traffic_prediction_with_missing_value/PEMS(M)/dataset/V_228.csv",
#                        "/home/wangao/Traffic_prediction_with_missing_value/METR-LA/dataset/metr.csv"]
# missing_rate = [4,6,8]
#
# for i in range(2):
#     for rate in missing_rate:
#         command = "python ./Scrips/generate_dataset_2direc_timelag.py --output_dir=\"{}\" --traffic_df_filename=\"{}\" --missing_rate={}"\
#             .format(output_dir[i], traffic_df_filename[i], rate)
#         os.system(command)
#         print('finish command')


output_dir = ["/home/wangao/Traffic_prediction_with_missing_value/PEMS(M)_NR_6/save/",
              "/home/wangao/Traffic_prediction_with_missing_value/METR-LA_NR_6/save/"]
horrizon = 6

traffic_df_filename = ["/home/wangao/Traffic_prediction_with_missing_value/PEMS(M)/dataset/V_228.csv",
                       "/home/wangao/Traffic_prediction_with_missing_value/METR-LA/dataset/metr.csv"]

mask_path = ["/PEMS(M)_NR/dataset/NR/", "/METR-LA_NR/dataset/NR/"]
missing_rate = [5,6,7,8]

root = "/home/wangao/Traffic_prediction_with_missing_value"

#生成NR的数据集
for i in range(1,2):
    for rate in missing_rate:
        temp_mask_path = root + mask_path[i] + "mask_{}.npy".format(rate)
        command = "python ./Scrips/generate_dataset_2direc_timelag_NR.py --output_dir=\"{}\" --traffic_df_filename=\"{}\" " \
                  "--missing_rate={} --mask_path=\"{}\" --horrizon={}"\
            .format(output_dir[i], traffic_df_filename[i], rate, temp_mask_path,horrizon)
        os.system(command)
        print('finish command')


#
# output_dir = ["/home/wangao/Traffic_prediction_with_missing_value/PEMS(M)_3/save/",
#               "/home/wangao/Traffic_prediction_with_missing_value/METR-LA_3/save/"]
#
# traffic_df_filename = ["/home/wangao/Traffic_prediction_with_missing_value/PEMS(M)/dataset/V_228.csv",
#                        "/home/wangao/Traffic_prediction_with_missing_value/METR-LA/dataset/metr.csv"]
#
# missing_rate = [8]
# # missing_rate = [2,3,4,5,6,7]
#
# root = "/home/wangao/Traffic_prediction_with_missing_value"
#
# for i in range(2):
#     for rate in missing_rate:
#         command = "python ./Scrips/generate_dataset_2direc_timelag.py --output_dir=\"{}\" --traffic_df_filename=\"{}\" " \
#                   "--missing_rate={} --horrizon=3"\
#             .format(output_dir[i], traffic_df_filename[i], rate)
#         os.system(command)

