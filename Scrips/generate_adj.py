import  numpy as np

data = np.load("../PEMS(M)/dataset/W_228_normalized.npy")
print(data.shape)
n = data.shape[0]

file_path = "../PEMS(M)/dataset/Adj(PEMS(M)).txt"
filr = open("../PEMS(M)/dataset/Adj(PEMS(M)).txt", "a")

for i in range(n):
    for j in range(n):
        filr.writelines("{} {} {:.6}\n".format(i,j,data[i][j]))

