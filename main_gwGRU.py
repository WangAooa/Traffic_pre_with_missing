import argparse
import os

import numpy as np
import torch
from torch.utils.data import *
import time
import csv
from pathlib import Path

from utils import *
# from model.rits import Model_1 as Model
from AE_model.gwnet_GRU import *

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")

torch.set_num_threads(1)

def generate_dataloader(train_dataset, test_dataset, args):

    dataset = Databidir(train_dataset['x'], train_dataset['x_mask'], train_dataset['y_missing'], train_dataset['y'],
                        train_dataset['time_lag'], train_dataset['time_lag_reverse'])
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    test_loader = DataLoader(
        dataset=Databidir(test_dataset['x'], test_dataset['x_mask'], test_dataset['y_missing'], test_dataset['y'],
                          test_dataset['time_lag'], test_dataset['time_lag_reverse']), batch_size=args.batch,
        shuffle=False, drop_last=True)

    scaler = StandardScaler(train_dataset['mean_std'][0], train_dataset['mean_std'][1])

    return train_loader, test_loader, scaler

def evaluate(net, test_loader, scaler):
    with torch.no_grad():
        net = net.eval()
        impute_mae_list, impute_mape_list, pre_mae_list, pre_mape_list, pre_mse_list = [], [], [], [], []
        #9_29
        only_im_mae_list, only_im_mape_list, only_im_mse_list = [], [], []
        start_time = time.time()
        for i, (x, x_mask, y_missing, y, time_lag,_) in enumerate(test_loader):
            (x, x_mask, y_missing, y, time_lag) = (transfer_to_device(x, device), transfer_to_device(x_mask, device),
                                                   transfer_to_device(y_missing, device), transfer_to_device(y, device),
                                                   transfer_to_device(time_lag, device))
            impute = net(x,x_mask, y_missing)

            impute_true = scaler.inverse_transform(impute)
            y_missing_true = scaler.inverse_transform(y_missing)

            impute_mae, impute_mape, _ = MAE(impute_true, x_mask, y_missing_true)

            #9_29
            #only_im_mae, only_im_mape = impute_MAE(impute_true, x_mask[:,3:,:,:], y_missing_true[:,3:,:,:])
            only_im_mae, only_im_mape, only_im_mse = impute_MAE(impute_true, x_mask, y_missing_true)
            only_im_mae_list.append(only_im_mae.item())
            only_im_mape_list.append(only_im_mape.item())
            only_im_mse_list.append(only_im_mse.item())

            impute_mae_list.append(impute_mae.item())
            impute_mape_list.append(impute_mape.item())

        return np.mean(only_im_mae_list),np.mean(only_im_mape_list),np.mean(only_im_mse_list), time.time() - start_time


def train(net, train_dataset, test_dataset,args):
    # create data loader
    train_loader, test_loader, scaler = generate_dataloader(train_dataset, test_dataset,args)

    # create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-8)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40],
    #                                                     gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    print('start training')

    Path(args.root_path + "/result/missing_rate_{}/{}/{}".format(args.missing_rate, args.dataset_type,
                                                                 args.model_type)).mkdir(parents=True, exist_ok=True)

    with open(args.root_path + '/result/missing_rate_{}/{}/{}/im_result.csv'.format(args.missing_rate, args.dataset_type, args.model_type), "a",
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['epoch' 'val_only_im_mae','val_only_im_mape','val_only_im_mse','time', 'inference_time'])

    earlystop = EarlyStopping(patience= 10,verbose=True,
                              path=args.root_path + '/result/missing_rate_{}/{}/{}/model.pt'.format(args.missing_rate, args.dataset_type,
                                                                                    args.model_type))
    # with open('./result/missing_rate_{}/result.txt'.format(missing_rate), 'a')as f:
    #     f.write('model type: {}\n'.format(model_type))
    min_im_mape = float('inf')

    for epoch_num in range(1000):
        start_time = time.time()
        print(epoch_num)

        net = net.train()
        impute_mae_list, impute_mape_list, pre_mae_list, pre_mape_list, pre_mse_list = [], [], [], [], []
        # 9_29
        only_impute_mae_list, only_impute_mape_list, only_impute_mse_list = [], [], []

        batch_seen = 0
        for i, (x, x_mask, y_missing, y, time_lag, _) in enumerate(train_loader):
            optimizer.zero_grad()
            (x, x_mask, y_missing, y, time_lag) = (transfer_to_device(x, device), transfer_to_device(x_mask, device),
                                                   transfer_to_device(y_missing, device), transfer_to_device(y, device),
                                                   transfer_to_device(time_lag, device))


            #impute= net(x,y_missing, batch_seen)  # B,T,N,F
            impute = net(x,x_mask,y_missing)

            # batch_seen = batch_seen + batch

            impute_true = scaler.inverse_transform(impute)
            y_missing_true = scaler.inverse_transform(y_missing)

            impute_mae, impute_mape, impute_mse = MAE(impute_true, x_mask, y_missing_true)

            only_impute_mae, only_impute_mape,_ = impute_MAE(impute_true, x_mask,y_missing_true)
            only_impute_mae_list.append(only_impute_mae.item())
            only_impute_mape_list.append(only_impute_mape.item())

            impute_mae_list.append(impute_mae.item())
            impute_mape_list.append(impute_mape.item())

            loss = impute_mae
            # loss = impute_mae
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

        lr_scheduler.step()

        end_time = time.time()

        # evaluate
        val_only_im_mae, val_only_im_mape,val_only_im_mse, inference_time = evaluate(net, test_loader, scaler)

        earlystop(val_only_im_mape, net.encoder_model)
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

        with open(args.root_path + '/result/missing_rate_{}/{}/{}/im_result.csv'.format(args.missing_rate, args.dataset_type, args.model_type), "a",
                  newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [epoch_num, val_only_im_mae, val_only_im_mape, val_only_im_mse, end_time - start_time, inference_time])

        # with open('./result/missing_rate_{}/result.txt'.format(missing_rate), 'a')as f:
        #     f.write(message + '\n')
    print('finish training')

def evaluate_pre(net, test_loader, scaler):
    with torch.no_grad():
        net = net.eval()
        impute_mae_list, impute_mape_list, pre_mae_list, pre_mape_list, pre_mse_list = [], [], [], [], []
        #9_29
        only_im_mae_list, only_im_mape_list = [], []
        start_time = time.time()
        for i, (x, x_mask, y_missing, y, time_lag,_) in enumerate(test_loader):
            (x, x_mask, y_missing, y, time_lag) = (transfer_to_device(x, device), transfer_to_device(x_mask, device),
                                                   transfer_to_device(y_missing, device), transfer_to_device(y, device),
                                                   transfer_to_device(time_lag, device))

            output = net(x,x_mask)

            pre_true = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)

            impute_mae, impute_mape, impute_mse = MAE(pre_true, x_mask, y_true)

            pre_mae_list.append(impute_mae.item())
            pre_mape_list.append(impute_mape.item())
            pre_mse_list.append(impute_mse.item())

        return np.mean(pre_mae_list),np.mean(pre_mape_list),np.mean(pre_mse_list), time.time() - start_time
def getparameters(epoch, model, optimizer):
    parameters_name = ["encoder_model.linear_dense_layer_2", "encoder_model.linear_dense_layer", "encoder_model.gwgru",
                       "encoder_model.gwgru_reverse","encoder_model.input_projection_2", "encoder_model.input_projection"]
    lr = 0.001
    if epoch % 5 ==0:
        lr = lr * (0.7**(epoch//5))
    fine_tune_step = [1,2,5,10,11]
    if epoch == 10:
        for name, para in model.named_parameters():
            if parameters_name[0] in name:
                para.requires_grad = True
        optimizer = torch.optim.Adam([
            {'params': model.decoder_model.parameters()},
            {'params': model.encoder_model.linear_dense_layer_2.parameters(), 'lr': lr/2.6}],
            lr=lr, eps=1e-8)
        # print_parameters(model.encoder_model)
    elif epoch == 11:
        for name, para in model.named_parameters():
            if parameters_name[1] in name:
                para.requires_grad = True
        optimizer = torch.optim.Adam([
            {'params': model.decoder_model.parameters()},
            {'params': model.encoder_model.linear_dense_layer_2.parameters(), 'lr': lr/2.6},
            {'params': model.encoder_model.linear_dense_layer.parameters(), 'lr': lr/(2.6**2)}],
            lr=lr, eps=1e-8)
        # print_parameters(model.encoder_model)
    elif epoch == 15:
        for name, para in model.named_parameters():
            if parameters_name[2] in name or parameters_name[3] in name:
                para.requires_grad = True
        optimizer = torch.optim.Adam([
            {'params': model.decoder_model.parameters()},
            {'params': model.encoder_model.linear_dense_layer_2.parameters(), 'lr': lr/2.6},
            {'params': model.encoder_model.linear_dense_layer.parameters(), 'lr': lr/(2.6**2)},
            {'params': model.encoder_model.gwgru.parameters(), 'lr': lr/(2.6**3)},
            {'params': model.encoder_model.gwgru_reverse.parameters(), 'lr': lr/(2.6**3)}],
            lr=lr, eps=1e-8)
        # print_parameters(model.encoder_model)
    elif epoch == 20:
        for name, para in model.named_parameters():
            if parameters_name[4] in name:
                para.requires_grad = True
        optimizer = torch.optim.Adam([
            {'params': model.decoder_model.parameters()},
            {'params': model.encoder_model.linear_dense_layer_2.parameters(), 'lr': lr / 2.6},
            {'params': model.encoder_model.linear_dense_layer.parameters(), 'lr': lr / (2.6 ** 2)},
            {'params': model.encoder_model.gwgru.parameters(), 'lr': lr / (2.6 ** 3)},
            {'params': model.encoder_model.gwgru_reverse.parameters(), 'lr': lr / (2.6 ** 3)},
            {'params': model.encoder_model.input_projection_2.parameters(), 'lr': lr/(2.6**4)}],
            lr=lr, eps=1e-8)
        # print_parameters(model.encoder_model)
    elif epoch == 21:
        for name, para in model.named_parameters():
            if parameters_name[5] in name:
                para.requires_grad = True
        optimizer = torch.optim.Adam([
            {'params': model.decoder_model.parameters()},
            {'params': model.encoder_model.linear_dense_layer_2.parameters(), 'lr': lr / 2.6},
            {'params': model.encoder_model.linear_dense_layer.parameters(), 'lr': lr / (2.6 ** 2)},
            {'params': model.encoder_model.gwgru.parameters(), 'lr': lr / (2.6 ** 3)},
            {'params': model.encoder_model.gwgru_reverse.parameters(), 'lr': lr / (2.6 ** 3)},
            {'params': model.encoder_model.input_projection_2.parameters(), 'lr': lr/(2.6**4)},
            {'params': model.encoder_model.input_projection.parameters(), 'lr': lr/(2.6**5)}],
            lr=lr, eps=1e-8)
        # print_parameters(model.encoder_model)
    return optimizer

def print_parameters(model):
    for para in model.parameters():
        print(para)
        print(para.requires_grad)
        print('________________')

def train_pre(net,train_dataset, test_dataset,args):
    train_loader, test_loader, scaler = generate_dataloader(train_dataset, test_dataset, args)

    # create the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-8)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40],
    #                                                     gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    print('start training')
    with open(args.root_path + '/result/missing_rate_{}/{}/{}/pre_result.csv'.format(args.missing_rate, args.dataset_type, args.model_type), "a",
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['epoch', 'val_pre_mae', 'val_pre_mape', 'val_pre_mse', 'time', 'inference_time'])

    earlystop = EarlyStopping(patience= 10, verbose=True,
                              path=args.root_path + '/result/missing_rate_{}/{}/{}/model_predict.pt'.format(args.missing_rate, args.dataset_type,
                                                                                    args.model_type))
    # with open('./result/missing_rate_{}/result.txt'.format(missing_rate), 'a')as f:
    #     f.write('model type: {}\n'.format(model_type))
    min_im_mape = float('inf')

    for epoch_num in range(1000):
        start_time = time.time()
        print(epoch_num)

        # fine tune
        # optimizer = getparameters(epoch_num,net, optimizer)

        net = net.train()
        impute_mae_list, impute_mape_list, pre_mae_list, pre_mape_list, pre_mse_list = [], [], [], [], []
        # 9_29
        only_impute_mae_list, only_impute_mape_list = [], []

        batch_seen = 0
        for i, (x, x_mask, y_missing, y, time_lag,_) in enumerate(train_loader):
            optimizer.zero_grad()
            (x, x_mask, y_missing, y, time_lag) = (transfer_to_device(x, device), transfer_to_device(x_mask, device),
                                                   transfer_to_device(y_missing, device), transfer_to_device(y, device),
                                                   transfer_to_device(time_lag, device))

            output = net(x, x_mask)
            # batch_seen = batch_seen + batch

            # print('y shape {}'.format(y.shape))
            # print('output shape {}'.format(output.shape))
            # output shape [T,B,N,F]
            pre_true = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)

            impute_mae, impute_mape, impute_mse = MAE(pre_true,x_mask, y_true)

            pre_mae_list.append(impute_mae.item())
            pre_mape_list.append(impute_mape.item())

            loss = impute_mae
            # loss = impute_mae
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

        lr_scheduler.step()

        end_time = time.time()
        # evaluate
        val_mae, val_mape, val_mse, inference_time = evaluate_pre(net,test_loader,scaler)

        earlystop(val_mape, net)
        if earlystop.early_stop:
            print("Early stopping")
            break



        # message = 'Epoch [{}/50] pre_mae:{:.4f}, pre_mape:{:.4f},' \
        #           'val_pre_mae:{:.4f}, val_pre_mape:{:.4f}, lr: {:.6f}, {:.1f}s' \
        #     .format(epoch_num, np.mean(pre_mae_list), np.mean(pre_mape_list), val_mae,val_mape,
        #             lr_scheduler.get_lr()[0], (end_time - start_time))
        # print(message)

        with open(args.root_path + '/result/missing_rate_{}/{}/{}/pre_result.csv'.format(args.missing_rate, args.dataset_type, args.model_type), "a",
                  newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [epoch_num, val_mae,val_mape, val_mse, end_time - start_time, inference_time])

        # with open('./result/missing_rate_{}/result.txt'.format(missing_rate), 'a')as f:
        #     f.write(message + '\n')
    print('finish training')

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
    parser.add_argument("--missing_rate", type=int, default=8)                 #
    # parser.add_argument("--adj_path", type=str, default="/nav-beijing/dataset/W_1362.npy")           #
    parser.add_argument("--adj_path", type=str, default="/PEMS(M)/dataset/W_228.npy")
    parser.add_argument("--dataset_type", type=str, default="PEMS(M)")                                      #
    # parser.add_argument("--model_type", type=str, default='gwnet_GRU_finetune_10')
    parser.add_argument("--model_type", type=str, default='gwnet_GRU')
    parser.add_argument("--root_path", type=str, default='/data/wangao')

    args = parser.parse_args()
    print(args)

    train_data_path = args.root_path + '/{}/save/{}/train_bidir.npz'.format(args.dataset_type,args.missing_rate)
    test_data_path = args.root_path + '/{}/save/{}/test_bidir.npz'.format(args.dataset_type,args.missing_rate)

    adj = np.load(args.root_path + args.adj_path)
    adj = torch.from_numpy(adj).to(device).to(torch.float)
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

    model_impute = model_impute.to(device)

    train_dataset = np.load(train_data_path)
    test_dataset = np.load(test_data_path)

    train(model_impute, train_dataset, test_dataset,args)

    print('finish training impute model')

    model_predict = Model(args.input_dim, adj, args.rnn_dim, args.num_node, num_rnn_layers=args.num_rnn_layer, output_dim=args.output_dim,
                         horrizon=args.horrizon,seq_len=args.seq_len).to(device)
    # print(torch.load(args.root_path + '/result/missing_rate_{}/{}/{}/model.pt'.format(args.missing_rate,args.dataset_type,args.model_type)))
    model_predict.encoder_model.load_state_dict(torch.load(args.root_path + '/result/missing_rate_{}/{}/{}/model.pt'.format(args.missing_rate,args.dataset_type,args.model_type)))

    ##freeze some layers
    for name, para in model_predict.named_parameters():
        if 'encoder_model' in name:
            para.requires_grad = False

    # for name, para in model_predict.named_parameters():
    #     # if(para.requires_grad):
    #     print(name)
    #     print(para)
    #     print(para.requires_grad)
    #     print('________________')
    # for para in model_predict.decoder_model.projection_layer_2.parameters():
    #     print(para)
    #     print(para.requires_grad)
    #     print('________________')

    train_pre(model_predict,train_dataset, test_dataset,args)
    #
    print("finish prediction step！")

    #test model
    command = "python model_test.py"
    for args_item in vars(args):
        if type(getattr(args, args_item)) is str:
            command = command + " --{}=\"{}\"".format(args_item, getattr(args, args_item))
        else:
            command = command + " --{}={}".format(args_item, getattr(args, args_item))

    os.system(command)

def test():
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
    parser.add_argument("--missing_rate", type=int, default=2)
    parser.add_argument("--adj_path", type=str, default="./PEMS(M)/dataset/W_228_normalized.npy")
    parser.add_argument("--dataset_type", type=str, default="PEMS(M)")
    parser.add_argument("--model_type", type=str, default="LSTM_gwnet")

    args = parser.parse_args()
    print("enter ")
    print(args)
    command = "python model_test.py"
    for args_item in vars(args):
        if type(getattr(args,args_item)) is str:
            command = command + " --{}=\"{}\"".format(args_item, getattr(args, args_item))
        else:
            command = command + " --{}={}".format(args_item,getattr(args,args_item))

    #command = command.replace(' ', '\ ').replace('(', '\(').replace(')', '\)')
    #print(command)
    os.system(command)
if __name__ == '__main__':
    main()
    #test()