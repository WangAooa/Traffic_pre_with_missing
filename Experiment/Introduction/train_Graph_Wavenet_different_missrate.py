import argparse
import os

import numpy as np
import torch
from torch.utils.data import *
import time
import csv
from pathlib import Path
import sys
import matplotlib.pyplot as plt
sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")

from utils import *
# from model.rits import Model_1 as Model
from Baseline_model.Prediction_model.Graph_Wavenet import gwnet as Model

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

torch.set_num_threads(1)

def generate_dataloader(train_dataset, test_dataset, args):

    dataset = Data(train_dataset['x'], train_dataset['y'])
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    test_loader = DataLoader(
        dataset=Data(test_dataset['x'], test_dataset['y']), batch_size=args.batch,shuffle=False, drop_last=True)

    scaler = StandardScaler(train_dataset['mean_std'][0], train_dataset['mean_std'][1])

    return train_loader, test_loader, scaler

def test_model(net, test_dataset, args):
    # create data loader
    test_loader = DataLoader(
        dataset=Data(test_dataset['x'], test_dataset['y']), batch_size=32, shuffle=False, drop_last=True)

    scaler = StandardScaler(test_dataset['mean_std'][0], test_dataset['mean_std'][1])

    # with torch.no_grad():
    #     net = net.eval()
    #     result_list = []
    #     for i, (x, y) in enumerate(test_loader):
    #         (x, y) = (transfer_to_device(x, device), transfer_to_device(y, device))
    #
    #         x = x.transpose(1,3)
    #         prediction = net(x)  # B,T,N,F
    #         prediction = prediction.transpose(1,3)
    #
    #         prediction_true = scaler.inverse_transform(prediction)
    #         y_true = scaler.inverse_transform(y)
    #
    #         result_list.append(prediction_true.reshape(args.horrizon, args.num_node).detach().cpu().numpy())
    # result_list = np.stack(result_list,0)
    # print(result_list.shape)
    #
    # np.save("./{}_{}_trainpart.npy".format(args.dataset_type, args.missing_rate),result_list)

    x_list,y_list= [],[]
    count = 0
    with torch.no_grad():
        net = net.eval()
        mae_list, mape_list, mse_list = [], [], []
        for i, (x, y) in enumerate(test_loader):
            (x, y) = (transfer_to_device(x, device), transfer_to_device(y, device))

            x = x.transpose(1, 3)
            prediction = net(x)  # B,T,N,F
            #prediction = prediction.transpose(1, 3)

            prediction_true = scaler.inverse_transform(prediction)
            y_true = scaler.inverse_transform(y)

            mae, mape, mse = masked_mae_loss(prediction_true, y_true)

            mae_list.append(mae.item())
            mape_list.append(mape.item())
            mse_list.append(mse.item())

            x_list.append(prediction_true.detach().cpu().numpy())
            y_list.append(y_true.detach().cpu().numpy())

        x_list = np.concatenate(x_list)
        y_list = np.concatenate(y_list)
        # plt.plot(x_list, label='prediction')
        # plt.plot(y_list, label = 'label')
        # plt.legend()
        # plt.savefig('./{}.png'.format(args.missing_rate))
        np.save('./{}_pre.npy'.format(args.missing_rate), np.array(x_list))
        np.save('./{}_label.npy'.format(args.missing_rate), np.array(y_list))

        # print("pre_mae :{}".format(np.mean(mae_list)))
        # print("pre_mape :{}".format(np.mean(mape_list)))
        # with open(args.root_path + "/result/missing_rate_{}/{}/{}/{}/test_result.txt".format(
        #         args.missing_rate, args.dataset_type, args.model_type, args.im_model_type), "a+") as f:
        #     f.writelines("pre_mae: {}\n".format(np.mean(mae_list)))
        #     f.writelines("pre_mape: {}\n".format(np.mean(mape_list)))
        #     f.writelines("pre_mse: {}\n".format(np.mean(mse_list)))


def evaluate(net, test_loader, scaler):
    with torch.no_grad():
        net = net.eval()
        mae_list, mape_list, mse_list = [], [], []

        start_time = time.time()
        for i, (x, y) in enumerate(test_loader):
            (x, y) = (transfer_to_device(x, device),  transfer_to_device(y, device))

            x = x.transpose(1, 3)
            prediction = net(x)  # B,T,N,F
            #prediction = prediction.transpose(1, 3)

            # batch_seen = batch_seen + batch

            prediction_true = scaler.inverse_transform(prediction)
            y_true = scaler.inverse_transform(y)

            mae, mape, mse = masked_mae_loss(prediction_true, y_true)

            mae_list.append(mae.item())
            mape_list.append(mape.item())
            mse_list.append(mse.item())

        return np.mean(mae_list),np.mean(mape_list),np.mean(mse_list),time.time() - start_time


def train(net, train_dataset, test_dataset,args):
    # create data loader
    train_loader, test_loader, scaler = generate_dataloader(train_dataset, test_dataset,args)

    # create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-8)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40],
    #                                                     gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    print('start training')

    earlystop = EarlyStopping(patience= 10,verbose=True,
                              path='./{}_{}_model.pt'.format(
                                   args.dataset_type,args.missing_rate))
    # with open('./result/missing_rate_{}/result.txt'.format(missing_rate), 'a')as f:
    #     f.write('model type: {}\n'.format(model_type))
    min_im_mape = float('inf')

    for epoch_num in range(1000):
        start_time = time.time()
        print(epoch_num)

        net = net.train()
        mae_list, mape_list, mse_list = [],[],[]

        batch_seen = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            (x,y) = (transfer_to_device(x, device), transfer_to_device(y, device))


            #impute= net(x,y_missing, batch_seen)  # B,T,N,F
            x = x.transpose(1, 3)
            prediction = net(x)  # B,T,N,F
            #prediction = prediction.transpose(1, 3)

            # batch_seen = batch_seen + batch

            prediction_true = scaler.inverse_transform(prediction)
            y_true = scaler.inverse_transform(y)

            mae, mape, mse= masked_mae_loss(prediction_true,y_true)

            mae_list.append(mae.item())
            mape_list.append(mape.item())
            mse_list.append(mse.item())

            loss = mae
            # loss = impute_mae
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

        lr_scheduler.step()
        end_time = time.time()
        # evaluate
        val_mae, val_mape,val_mse,inference_time = evaluate(net, test_loader, scaler)

        earlystop(val_mape, net)
        if earlystop.early_stop:
            print("Early stopping")
            break
    print('finish training')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--rnn_dim", type=int, default=64)
    parser.add_argument("--num_node", type=int, default=228)                #
    parser.add_argument("--num_rnn_layer", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--horrizon", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--missing_rate", type=int, default=2)                 #
    parser.add_argument("--adj_path", type=str, default="/PEMS(M)/dataset/W_228_normalized.npy")           #
    parser.add_argument("--dataset_type", type=str, default="PEMS(M)_NR")                                      #
    parser.add_argument("--model_type", type=str, default='Graph_Wavenet')
    parser.add_argument("--root_path", type=str, default="/home/wangao/Traffic_prediction_with_missing_value")
    parser.add_argument("--im_model_type", type=str, default='brits')

    args = parser.parse_args()
    print(args)

    train_data_path =  './{}_{}_train_pretask.npz'.format(args.dataset_type,args.missing_rate)
    test_data_path = './{}_{}_test_pretask.npz'.format(args.dataset_type,args.missing_rate)

    adj = np.load(args.root_path +args.adj_path)
    adj = torch.from_numpy(adj).to(device).to(torch.float)
    #print(adj[:5, :5])

    # adj = [adj]
    #adj.append(torch.from_numpy(np.load(adj_reverse_path)).to(device).to(torch.float))
    # adj = torch.eye(num_node).to(device)

    #model = Model(input_dim, rnn_dim, batch, num_node, time_step, pre_step, adj)
    model = Model(device, num_nodes=args.num_node, supports=[adj], in_dim=args.input_dim)
    #model = Model(num_node, supports=adj)
    # print(device)
    # print(model_impute)

    model = model.to(device)

    # model.load_state_dict(torch.load(
    #     './{}_{}_model.pt'.format(args.dataset_type, args.missing_rate)))
    train_dataset = np.load(train_data_path)
    test_dataset = np.load(test_data_path)

    train(model, train_dataset, test_dataset,args)

    # test model performance

    test_data_path = './{}_{}_train_pretask.npz'.format(args.dataset_type,args.missing_rate)
    test_dataset = np.load(test_data_path)
    test_model(model,test_dataset, args)

if __name__ == '__main__':
    main()
    #test()