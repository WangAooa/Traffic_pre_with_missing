import argparse

import numpy as np
import torch
from torch.utils.data import *
import time
import csv
from pathlib import Path

from utils import *
# from model.rits import Model_1 as Model
from BRITS_model.brits import brits as Model

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")

torch.set_num_threads(1)

def evaluate(net, test_loader, scaler):
    with torch.no_grad():
        net = net.eval()
        impute_mae_list, impute_mape_list, pre_mae_list, pre_mape_list, pre_mse_list = [], [], [], [], []

        only_im_mae_list, only_im_mape_list, only_im_mse_list = [], [], []
        start_time = time.time()
        for i, (x, x_mask, y_missing, y, time_lag, time_lag_reverse) in enumerate(test_loader):
            (x, x_mask, y_missing, y, time_lag, time_lag_reverse) = (transfer_to_device(x, device), transfer_to_device(x_mask, device),
                                                   transfer_to_device(y_missing, device), transfer_to_device(y, device),
                                                   transfer_to_device(time_lag, device), transfer_to_device(time_lag_reverse, device))

            impute, predition = net(x, x_mask, time_lag, time_lag_reverse)

            impute_true = scaler.inverse_transform(impute)
            y_missing_true = scaler.inverse_transform(y_missing)

            predition_true = scaler.inverse_transform(predition)
            y_true = scaler.inverse_transform(y)

            pre_mae, pre_mape, pre_mse = masked_mae_loss(predition_true, y_true)

            only_im_mae, only_im_mape, only_im_mse = impute_MAE(impute_true, x_mask, y_missing_true)
            only_im_mae_list.append(only_im_mae.item())
            only_im_mape_list.append(only_im_mape.item())
            only_im_mse_list.append(only_im_mse.item())

            pre_mae_list.append(pre_mae.item())
            pre_mape_list.append(pre_mape.item())
            pre_mse_list.append(pre_mse.item())

        return np.mean(only_im_mae_list),np.mean(only_im_mape_list),np.mean(only_im_mse_list), \
               np.mean(pre_mae_list), np.mean(pre_mape_list), np.mean(pre_mse_list), time.time() -start_time


def train(net, train_dataset, test_dataset, args):
    # create data loader
    dataset = Databidir(train_dataset['x'], train_dataset['x_mask'], train_dataset['y_missing'], train_dataset['y'],
                   train_dataset['time_lag'], train_dataset['time_lag_reverse'])
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    test_loader = DataLoader(
        dataset=Databidir(test_dataset['x'], test_dataset['x_mask'], test_dataset['y_missing'], test_dataset['y'],
                     test_dataset['time_lag'], test_dataset['time_lag_reverse']), batch_size=args.batch, shuffle=False, drop_last=True)

    scaler = StandardScaler(train_dataset['mean_std'][0], train_dataset['mean_std'][1])

    # create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-8)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40],
    #                                                     gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # print('start training')

    Path(args.root_path +'/result/missing_rate_{}/{}/{}'.format(args.missing_rate, args.dataset_type, args.model_type)).mkdir(parents=True, exist_ok=True)

    with open(args.root_path +'/result/missing_rate_{}/{}/{}/result.csv'.format(args.missing_rate, args.dataset_type, args.model_type), "a",
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['epoch', 'val_only_im_mae','val_only_im_mape','val_only_im_mse', 'val_pre_mae', 'val_pre_mape',
             'val_pre_mse', 'time', 'inference_time'])

    earlystop = EarlyStopping(patience= 10,verbose=True,
                              path=args.root_path +'/result/missing_rate_{}/{}/{}/model.pt'.format(args.missing_rate, args.dataset_type,
                                                                                    args.model_type))
    # with open('./result/missing_rate_{}/result.txt'.format(missing_rate), 'a')as f:
    #     f.write('model type: {}\n'.format(model_type))
    min_val_mae = float('inf')

    for epoch_num in range(1000):
        start_time = time.time()
        #print(epoch_num)

        net = net.train()
        impute_mae_list, impute_mape_list, pre_mae_list, pre_mape_list, pre_mse_list = [], [], [], [], []

        only_impute_mae_list, only_impute_mape_list, only_impute_mse_list = [], [], []
        for i, (x, x_mask, y_missing, y, time_lag, time_lag_reverse) in enumerate(train_loader):
            optimizer.zero_grad()
            (x, x_mask, y_missing, y, time_lag, time_lag_reverse) = (transfer_to_device(x, device), transfer_to_device(x_mask, device),
                                                   transfer_to_device(y_missing, device), transfer_to_device(y, device),
                                                   transfer_to_device(time_lag, device), transfer_to_device(time_lag_reverse, device))

            impute, predition = net(x, x_mask, time_lag, time_lag_reverse)  # B,T,N,F

            impute_true = scaler.inverse_transform(impute)
            y_missing_true = scaler.inverse_transform(y_missing)

            predition_true = scaler.inverse_transform(predition)
            y_true = scaler.inverse_transform(y)

            pre_mae, pre_mape, pre_mse = masked_mae_loss(predition_true, y_true)

            only_impute_mae, only_impute_mape, only_impute_mse = impute_MAE(impute_true, x_mask, y_missing_true)
            only_impute_mae_list.append(only_impute_mae.item())
            only_impute_mape_list.append(only_impute_mape.item())
            only_impute_mse_list.append(only_impute_mse.item())
            # if impute_mape  != impute_mape :
            #     print('nan {}'.format(i))
            #     torch.save(impute_true, './impute_true.pt')
            #     torch.save(y_missing_true, './y_missing_true.pt')
            #     torch.save(x_mask, './x_mask.pt')
            #     break

            pre_mae_list.append(pre_mae.item())
            pre_mape_list.append(pre_mape.item())
            pre_mse_list.append(pre_mse.item())

            loss = only_impute_mae + pre_mae
            # loss = impute_mae
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

        lr_scheduler.step()

        end_time = time.time()
        # evaluate
        val_only_im_mae, val_only_im_mape, val_only_im_mse, val_pre_mae, \
        val_pre_mape, val_pre_mse, inference_time = evaluate(net, test_loader, scaler)

        # store model
        # if val_pre_mape < min_val_mae:
        #     torch.save(net.state_dict(), './result/missing_rate_{}/{}/{}/model.pt'.format(missing_rate,dataset_type,model_type))
        #     min_val_mae = val_pre_mape
        #     print('save model')
        earlystop(val_only_im_mape, net)
        if earlystop.early_stop:
            print("Early stopping")
            break



        # message = 'Epoch [{}/50] only_im_mae:{:.4f}, only_im_mape:{:.4f},  predition_mae: {:.4f}, ' \
        #           'prediction_mape: {},val_only_im_mae:{:.4f}, val_only_im_mape:{:.4f}, ' \
        #           'val_pre_mae: {:.4f}, val_pre_mape: {}, lr: {:.6f}, {:.1f}s' \
        #     .format(epoch_num, np.mean(only_impute_mape_list), np.mean(only_impute_mape_list), np.mean(pre_mae_list),
        #             np.mean(pre_mape_list), val_only_im_mae, val_only_im_mape, val_pre_mae, val_pre_mape,
        #             lr_scheduler.get_lr()[0], (end_time - start_time))
        # print(message)

        with open(args.root_path +'/result/missing_rate_{}/{}/{}/result.csv'.format(args.missing_rate, args.dataset_type, args.model_type), "a",
                  newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [epoch_num, val_only_im_mae, val_only_im_mape, val_only_im_mse,
                  val_pre_mae, val_pre_mape,val_pre_mse, end_time - start_time, inference_time])

        # with open('./result/missing_rate_{}/result.txt'.format(missing_rate), 'a')as f:
        #     f.write(message + '\n')
    # print('finish training')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--rnn_dim", type=int, default=64)
    parser.add_argument("--num_node", type=int, default=1362)
    parser.add_argument("--num_rnn_layer", type=int, default=2)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--horrizon", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--missing_rate", type=int, default=8)
    parser.add_argument("--adj_path", type=str, default="/data/wangao/nav-beijing/dataset/W_1362.npy")
    parser.add_argument("--dataset_type", type=str, default="nav-beijing")
    parser.add_argument("--model_type", type=str, default='brits')
    parser.add_argument("--root_path", type=str, default="/data/wangao")

    args = parser.parse_args()

    train_data_path =args.root_path + '/{}/save/{}/train_bidir.npz'.format(args.dataset_type,args.missing_rate)
    test_data_path =args.root_path + '/{}/save/{}/test_bidir.npz'.format(args.dataset_type,args.missing_rate)

    # adj = np.load(args.adj_path)
    # adj = torch.from_numpy(adj).to(device).to(torch.float)
    # print(adj[:5, :5])

    # adj = [adj]
    #adj.append(torch.from_numpy(np.load(adj_reverse_path)).to(device).to(torch.float))
    # adj = torch.eye(num_node).to(device)

    model = Model(args.input_dim, args.rnn_dim, args.batch, args.num_node, args.seq_len, args.horrizon)
    # print(device)
    # print(model)
    model = model.to(device)

    train_dataset = np.load(train_data_path)
    test_dataset = np.load(test_data_path)

    train(model, train_dataset, test_dataset, args)



if __name__ == '__main__':
    main()