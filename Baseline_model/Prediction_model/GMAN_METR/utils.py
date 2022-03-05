import numpy as np
import pandas as pd

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y
def generate_TE(num_step):
    TE = np.ones((num_step,2))
    for i in range(num_step):
        x, y = (i // 288) % 7, i % 288
        TE[i] = [x,y]
    return TE

def generate_real_missing_data(im_data,origin_data, his_len, pre_len):
    length, n_route = origin_data.shape[0], origin_data.shape[1]

    batch_size = length - his_len - pre_len + 1
    x, y_label = [], []

    for i in range(batch_size):
        # x_t = im_data[i: i + his_len] * origin_data[i : i + his_len]
        x_t = im_data[i: i + his_len]
        # x_t[mask_t == 0] = 0
        y_t = origin_data[i + his_len: i + his_len + pre_len]

        x.append(x_t)
        y_label.append(y_t)

    x = np.stack(x, 0)
    y = np.stack(y_label, 0)
    return x, y

def loadData(args):
    # Traffic
    df = np.load(args.traffic_file)
    origin_df =  pd.read_csv(args.full_traffic_file, header=None).values.astype(float)

    #Traffic = df.values
    Traffic = df
    # train/val/test

    mean, std = np.mean(origin_df), np.std(origin_df)
    origin_df = (origin_df - mean) / std

    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]

    origin_train = origin_df[:train_steps]
    origin_val = origin_df[train_steps : train_steps + val_steps]
    origin_test = origin_df[-test_steps :]
    # X, Y 
    # trainX, trainY = seq2instance(train, args.P, args.Q)
    # valX, valY = seq2instance(val, args.P, args.Q)
    # testX, testY = seq2instance(test, args.P, args.Q)
    trainX, trainY = generate_real_missing_data(train,origin_train, args.P, args.Q)
    valX, valY = generate_real_missing_data(val,origin_val, args.P, args.Q)
    testX, testY = generate_real_missing_data(test,origin_test, args.P, args.Q)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # spatial embedding 
    f = open(args.SE_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]
        
    # temporal embedding 
    # Time = df.index
    # dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    # timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
    #             // Time.freq.delta.total_seconds()
    # timeofday = np.reshape(timeofday, newshape = (-1, 1))
    # Time = np.concatenate((dayofweek, timeofday), axis = -1)
    Time = generate_TE(num_step)

    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)
    
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)
