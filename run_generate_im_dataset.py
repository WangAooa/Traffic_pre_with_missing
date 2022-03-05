import os

missing_rate = [2,3,4,5,6,7,8]
# missing_rate = [3,5,7]

vector_datapath = ["/PEMS(M)/dataset/V_228.csv", "/METR-LA/dataset/metr.csv"]
# dataset_type = ["PEMS(M)_NR", "METR-LA_NR"]
dataset_type = ["PEMS(M)", "METR-LA"]
model_type = ["BTMF_r10","BTMF"]
rank = [10,50]
# missing_rate = [8]
# vector_datapath = ["/METR-LA/dataset/metr.csv"]
# dataset_type = ["METR-LA_NR"]
# model_type = ["BTMF"]
# rank = [50]

#BTMF
# for miss_rate in missing_rate:
#     for i in range(len(vector_datapath)):
#         for r in range(len(rank)):
#             command = "python ./Baseline_script/BTMF_test.py  --missing_rate={} --vector_datapath=\"{}\" --dataset_type=\"{}\" " \
#                       "--model_type={} --rank={}"\
#                 .format( miss_rate, vector_datapath[i],dataset_type[i],model_type[r],rank[r])
#             os.system(command)

#brits
# num_node = [228,207]
# for miss_rate in missing_rate:
#     for i in range(len(vector_datapath)):
#         command = "python ./Baseline_script/brits_test.py  --num_node={} --missing_rate={} --vector_datapath=\"{}\" --dataset_type=\"{}\"" \
#             .format(num_node[i], miss_rate, vector_datapath[i],dataset_type[i])
#         os.system(command)
