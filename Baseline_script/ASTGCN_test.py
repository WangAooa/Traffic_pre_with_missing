import argparse
import os

import numpy as np
import torch
from torch.utils.data import *
import time
import csv
from pathlib import Path
import sys

sys.path.append("/home/wangao/Traffic_prediction_with_missing_value")

from utils import *
# from model.rits import Model_1 as Model
from Baseline_model.Prediction_model.ASTGCN import *

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
        dataset=Data(test_dataset['x'], test_dataset['y']), batch_size=args.batch, shuffle=False, drop_last=True)

    scaler = StandardScaler(test_dataset['mean_std'][0], test_dataset['mean_std'][1])

    result = []

    with torch.no_grad():
        net = net.eval()
        mae_list, mape_list,mse_list = [], [], []
        for i, (x, y) in enumerate(test_loader):
            (x, y) = (transfer_to_device(x, device), transfer_to_device(y, device))

            x = x.permute(0, 2, 3, 1)
            prediction = net(x)  # B,T,N,F
            #prediction = prediction.transpose(1,3)

            prediction_true = scaler.inverse_transform(prediction)

            print(i)
            prediction_true = prediction_true.cpu().numpy()
            print(prediction_true.shape)
            print(prediction_true[:,-1,0,0].shape)
            result.append(prediction_true[:,-1,0,0])
        result = np.array(result).reshape(-1)
        print(result.shape)
        np.save('/home/wangao/Traffic_prediction_with_missing_value/Experiment/prediction_result/ASTGCN_PEMS(M)_NR_8.npy',result)
        #     y_true = scaler.inverse_transform(y)
        #
        #     mae, mape, mse = masked_mae_loss(prediction_true, y_true)
        #
        #     mae_list.append(mae.item())
        #     mape_list.append(mape.item())
        #     mse_list.append(mse.item())
        #
        # # print("pre_mae :{}".format(np.mean(pre_mae_list)))
        # # print("pre_mape :{}".format(np.mean(pre_mape_list)))
        # with open(args.root_path +"/result/missing_rate_{}/{}/{}/{}/test_result.txt".format(
        #         args.missing_rate,args.dataset_type,args.model_type, args.im_model_type),"a+") as f:
        #
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

            x = x.permute(0, 2, 3, 1)
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

    Path(args.root_path + "/result/missing_rate_{}/{}/{}/{}".format(args.missing_rate, args.dataset_type,
                                                                 args.model_type, args.im_model_type)).mkdir(parents=True, exist_ok=True)

    with open(args.root_path +'/result/missing_rate_{}/{}/{}/{}/result.csv'.format(
            args.missing_rate,args.dataset_type, args.model_type, args.im_model_type), "a",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['epoch' 'val_only_pre_mae','val_only_pre_mape','val_only_pre_mse','time','inference_time'])

    earlystop = EarlyStopping(patience= 7,verbose=True,
                              path=args.root_path +'/result/missing_rate_{}/{}/{}/{}/model.pt'.format(
                                  args.missing_rate, args.dataset_type,args.model_type,args.im_model_type))
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
            x = x.permute(0,2,3,1)
            prediction = net(x)  # B,T,N,F
            #prediction = prediction.transpose(1, 3)

            # batch_seen = batch_seen + batch
            # print('prediction_shape: {}'.format(prediction.shape))
            # print('label shape: {}'.format(y.shape))

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


        # message = 'Epoch [{}/50] only_im_mae:{:.4f}, only_im_mape:{:.4f}, impute_mae: {:.4f}, impute_mape: {},' \
        #           'val_only_im_mae:{:.4f}, val_only_im_mape:{:.4f},val_impute_mae: {:.4f},val_impute_mape: {}, ' \
        #            'lr: {:.6f}, {:.1f}s' \
        #     .format(epoch_num, np.mean(only_impute_mape_list), np.mean(only_impute_mape_list), np.mean(impute_mae_list), np.mean(impute_mape_list),
        #              val_only_im_mae, val_only_im_mape,val_impute_mae, val_impute_mape,
        #             lr_scheduler.get_lr()[0], (end_time - start_time))
        # print(message)

        with open(args.root_path +'/result/missing_rate_{}/{}/{}/{}/result.csv'.format(
            args.missing_rate,args.dataset_type, args.model_type, args.im_model_type), "a",
                  newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [epoch_num, val_mae, val_mape, val_mse, end_time - start_time,inference_time])

        # with open('./result/missing_rate_{}/result.txt'.format(missing_rate), 'a')as f:
        #     f.write(message + '\n')
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
    parser.add_argument("--missing_rate", type=int, default=8)                 #
    parser.add_argument("--adj_path", type=str, default="/PEMS(M)/dataset/W_228_normalized.npy")           #
    parser.add_argument("--dataset_type", type=str, default="PEMS(M)")                                      #
    parser.add_argument("--model_type", type=str, default='ASTGCN_pastweek')
    parser.add_argument("--root_path", type=str, default="/data/wangao")
    parser.add_argument("--im_model_type", type=str, default='brits')

    args = parser.parse_args()
    print(args)

    train_data_path = args.root_path + '/{}/save/{}/{}_train_pastweek_pretask.npz'.format(args.dataset_type,args.missing_rate,args.im_model_type)
    test_data_path = args.root_path +'/{}/save/{}/{}_test_pastweek_pretask.npz'.format(args.dataset_type,args.missing_rate,args.im_model_type)

    adj = np.load(args.root_path +args.adj_path)
    #adj = torch.from_numpy(adj).to(device).to(torch.float)
    #print(adj[:5, :5])

    # adj = [adj]
    #adj.append(torch.from_numpy(np.load(adj_reverse_path)).to(device).to(torch.float))
    # adj = torch.eye(num_node).to(device)

    #model = Model(input_dim, rnn_dim, batch, num_node, time_step, pre_step, adj)
    model = Model(device,2,1, 3, 64,64,1, adj,args.horrizon,args.seq_len,args.num_node)
    #model = Model(num_node, supports=adj)
    # print(device)
    # print(model_impute)

    model = model.to(device)

    train_dataset = np.load(train_data_path)
    test_dataset = np.load(test_data_path)

    # train(model, train_dataset, test_dataset,args)

    # test model performance

    test_data_path = args.root_path +'/{}/save/{}/{}_val_pastweek_pretask.npz'.format(args.dataset_type,args.missing_rate,args.im_model_type)
    test_dataset = np.load(test_data_path)
    # test_model(model,test_dataset, args)
    test_model(model, train_dataset, args)

if __name__ == '__main__':
    main()
    #test()