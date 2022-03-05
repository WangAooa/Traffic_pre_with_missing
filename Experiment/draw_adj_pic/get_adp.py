import argparse
import os

import numpy as np
import torch
from torch.utils.data import *
import time
import csv
from pathlib import Path
import sys
from matplotlib import pyplot as plt

sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")

# from model.rits import Model_1 as Model
from AE_model.gwnet_GRU import *

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")

torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    # parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--rnn_dim", type=int, default=64)
    # parser.add_argument("--rnn_dim", type=int, default=128)
    # parser.add_argument("--num_node", type=int, default=1362)                #
    parser.add_argument("--num_node", type=int, default=228)
    parser.add_argument("--num_rnn_layer", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--horrizon", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--missing_rate", type=int, default=2)                 #
    # parser.add_argument("--adj_path", type=str, default="/nav-beijing/dataset/W_1362.npy")           #
    parser.add_argument("--adj_path", type=str, default="/PEMS(M)/dataset/W_228.npy")
    parser.add_argument("--dataset_type", type=str, default="PEMS(M)_NR")                                      #
    # parser.add_argument("--model_type", type=str, default='gwnet_GRU_finetune_10')
    parser.add_argument("--model_type", type=str, default='gwnet_GRU')
    parser.add_argument("--root_path", type=str, default='/data/wangao')

    args = parser.parse_args()
    print(args)


    adj = np.load(args.root_path + args.adj_path)
    #print(adj[:5, :5])
    adj = [adj]
    #adj.append(torch.from_numpy(np.load(adj_reverse_path)).to(device).to(torch.float))
    # adj = torch.eye(num_node).to(device)

    #model = Model(input_dim, rnn_dim, batch, num_node, time_step, pre_step, adj)
    model_impute = Model(args.input_dim, adj, args.rnn_dim, args.num_node, num_rnn_layers=args.num_rnn_layer, output_dim=args.output_dim,
                         horrizon=args.seq_len,seq_len=args.seq_len)
    #model = Model(num_node, supports=adj)
    # print(device)
    # print(model_impute)
    for name, para in model_impute.named_parameters():
        # if(para.requires_grad):
        # print(name)
        # print(para.shape)
        # print(para.requires_grad)
        # print('________________')
        if name == "encoder_model.gwgru.0.gcn.nodevec1":
            print("get")
            vec_1 = para
        if name == "encoder_model.gwgru.0.gcn.nodevec2":
            vec_2 = para

    # print(torch.load(args.root_path + '/result/missing_rate_{}/{}/{}/model.pt'.format(args.missing_rate,args.dataset_type,args.model_type)))
    model_impute.load_state_dict(torch.load(args.root_path + '/result/missing_rate_{}/{}/{}/model_predict.pt'.format(
        args.missing_rate, args.dataset_type, args.model_type)))

    adp = F.softmax(F.relu(torch.mm(vec_1,vec_2)), dim=1)
    adp = adp.detach().cpu().numpy()
    print(type(adp))
    print(adp[:5,:5])

    fig = plt.figure(figsize=(6, 6), dpi=150)
    plt.cla()
    # cmap = plt.cm.Spectral
    cmap = plt.cm.Oranges
    pic_fontsize = 14
    # t = plt.imshow(dori, vmin=0, vmax=90)
    t = plt.imshow(adp, cmap=cmap, vmin=0, vmax=0.05)
    # cbar = plt.colorbar(t, fraction=0.045, pad=0.1)
    cbar = plt.colorbar(t, fraction=0.045, pad=0.01)
    cbar.ax.tick_params(labelsize=pic_fontsize)
    # cbar.ax.set_ylabel('speed miles/h', fontsize=pic_fontsize)
    plt.xticks(fontsize=pic_fontsize)
    plt.yticks(fontsize=pic_fontsize)
    # plt.xlabel('Time', fontsize=pic_fontsize)
    # plt.ylabel('Station No.', fontsize=pic_fontsize)
    # fig.savefig('testpic/' + 'test_original.png')
    # plt.show()
    plt.savefig('./adp_PEMS.pdf')

    fig = plt.figure(figsize=(6, 6), dpi=150)
    plt.cla()
    # cmap = plt.cm.Spectral
    cmap = plt.cm.Oranges
    pic_fontsize = 14
    # t = plt.imshow(dori, vmin=0, vmax=90)
    t = plt.imshow(adj[0], cmap=cmap, vmin=0, vmax=0.05)
    # cbar = plt.colorbar(t, fraction=0.045, pad=0.1)
    cbar = plt.colorbar(t, fraction=0.045, pad=0.01)
    cbar.ax.tick_params(labelsize=pic_fontsize)
    # cbar.ax.set_ylabel('speed miles/h', fontsize=pic_fontsize)
    plt.xticks(fontsize=pic_fontsize)
    plt.yticks(fontsize=pic_fontsize)
    # plt.xlabel('Time', fontsize=pic_fontsize)
    # plt.ylabel('Station No.', fontsize=pic_fontsize)
    # fig.savefig('testpic/' + 'test_original.png')
    # plt.show()
    plt.savefig('./adj_PEMS.pdf')



if __name__ == '__main__':
    main()
    #test()