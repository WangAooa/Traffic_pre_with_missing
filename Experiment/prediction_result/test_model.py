import argparse

import numpy as np
import torch
from torch.utils.data import *
import time
import csv
import sys
sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")

from utils import *
from AE_model.gwnet_GRU import Model


# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")

torch.set_num_threads(1)

def test_model(net, args):
    # create data loader
    origin_data = pd.read_csv('/data/wangao/PEMS(M)/dataset/V_228.csv', header=None).values.astype(float)
    print(origin_data.shape)
    mask = np.load('/data/wangao/PEMS(M)_NR/dataset/NR/mask_8.npy')

    scaler = StandardScaler(origin_data.mean(), origin_data.std())
    origin_data = scaler.transform(origin_data)


    pre_result = []

    with torch.no_grad():
        net = net.eval()
        for i in range(origin_data.shape[0] - 12 + 1):
            x = origin_data[i : i+12] * mask[i : i+12]
            x_mask = mask[i : i+12]

            x, x_mask = torch.from_numpy(x).reshape((1,12,228,1)).to(torch.float32).to(device), torch.from_numpy(x_mask).reshape((1,12,228,1)).to(torch.float32).to(device)

            predition = net(x, x_mask)  # B,T,N,F

            predition_true = scaler.inverse_transform(predition)

            pre_result.append(predition_true[0,-1,0,0].cpu().numpy())
        result = np.array(pre_result)
        print(result.shape)
        print(result[:10])
        np.save('./{}_{}_{}.npy'.format(args.model_type, args.dataset_type, args.missing_rate), result)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--rnn_dim", type=int, default=64)
    parser.add_argument("--num_node", type=int, default=228)
    parser.add_argument("--num_rnn_layer", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--horrizon", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--missing_rate", type=int, default=8)
    parser.add_argument("--adj_path", type=str, default="/PEMS(M)/dataset/W_228_normalized.npy")
    parser.add_argument("--dataset_type", type=str, default="PEMS(M)_NR")
    parser.add_argument("--model_type", type=str, default='gwnet_GRU_rnn_layer_3')
    parser.add_argument("--root_path", type=str, default='/data/wangao')

    args = parser.parse_args()
    # for args_item in vars(args):
    #     print(args_item)
    #     print(getattr(args,args_item))
    print("model test")
    print(args)


    adj = np.load(args.root_path +args.adj_path)
    adj = torch.from_numpy(adj).to(device).to(torch.float)
    # print(adj[:5, :5])

    adj = [adj]
    # adj.append(torch.from_numpy(np.load(adj_reverse_path)).to(device).to(torch.float))
    # adj = torch.eye(num_node).to(device)

    model = Model(args.input_dim, adj, args.rnn_dim, args.num_node, num_rnn_layers=args.num_rnn_layer, output_dim=args.output_dim,
                  horrizon=args.horrizon,seq_len=args.seq_len)
    # print(device)
    # print(model)
    model = model.to(device)
    # model.load_state_dict(torch.load('./result/missing_rate_{}/{}/{}/model_predict.pt'.format(
    #     args.missing_rate,args.dataset_type,args.model_type),  map_location='cpu'))
    model.load_state_dict(torch.load(args.root_path +'/result/missing_rate_{}/{}/{}/model_predict.pt'.format(
        args.missing_rate, args.dataset_type, args.model_type)))
    # train_dataset = np.load(train_data_path)
    # test_dataset = np.load(test_data_path)

    test_model(model, args)


if __name__ == '__main__':
    main()