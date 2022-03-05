import os

missing_rate = [2,8]
for i in missing_rate:
    command = "python train_Graph_Wavenet_different_missrate.py --missing_rate={}".format(i)
    os.system(command)