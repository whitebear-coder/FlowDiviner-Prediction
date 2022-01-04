"""project:lstm-pytorch-3.0
author:lin zexu
add:preprocessing:扩展窗口拆分器
"""
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
import math


def wavelet_denoise(input):
    '''
    小波变换及重构实现降噪
    :param input: 输入序列
    :return: 返回降噪后的序列
    '''
    # 小波函数取db4
    db4 = pywt.Wavelet('db4')
    # 分解
    coeffs = pywt.wavedec(input, db4)
    # 高频系数
    coeffs[len(coeffs) - 1] *= 0
    coeffs[len(coeffs) - 2] *= 0
    # 重构
    meta = pywt.waverec(coeffs, db4)
    return meta


def smooth(data):
    return np.log1p(data)


def min_max_scaler(data, k=0):
    sc = MinMaxScaler(feature_range=(0, 1))
    data_scaled = sc.fit_transform(data) + 0.001
    return data_scaled


# 引入数据
def load_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data)
    data = data.ravel()
    data = wavelet_denoise(data)
    data = data.reshape(-1, 1)
    data = smooth(data)
    # print(data.shape)
    # data = min_max_scaler(data)
    # print(data.shape)
    series = []
    for i in range(len(data)):
        series.append(list(map(float, data[i])))  # 将数据转化为int类型
    return series


def create_datasets(datasets, look_back, look_head):
    data_x = []
    data_y = []
    for i in range(len(datasets) - look_back - look_head + 1):
        window = datasets[i:(i + look_back)]
        data_x.append(window)
        data_y.append(datasets[i + look_back:i + look_back + look_head])
    return np.array(data_x), np.array(data_y)


def mape(y_pred, y_true):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def split_k(datasets, k, i):
    train_size = int(len(datasets) * (1 / k))
    return datasets[int(train_size * i):int(train_size) * (i+1)]


# 训练集和测试集分离
def train_test_split(datasets, train_spt, test_spt):
    train_size = int(len(datasets) * train_spt)
    test_size = int(len(datasets) * test_spt)
    return datasets[:train_size], datasets[train_size:(train_size + test_size)], datasets[(train_size + test_size):]


def rev_trans(preds, labels):
    sc = MinMaxScaler(feature_range=(0, 1))
    # 反归一化
    pred_set_end = sc.inverse_transform(preds)
    labels_set_end = sc.inverse_transform(labels)
    # 反平滑
    pred_set_end = np.expm1(pred_set_end)
    labels_set_end = np.expm1(labels_set_end)
    return pred_set_end, labels_set_end


def evaluate(pred_set_end, labels_set_end):
    mape1 = mape(pred_set_end, labels_set_end)
    rmse = math.sqrt(metrics.mean_squared_error(pred_set_end, labels_set_end))
    mae = metrics.mean_absolute_error(pred_set_end, labels_set_end)

    print('均方根误差: %.6f' % rmse)
    print('平均绝对误差: %.6f' % mae)
    print('平均百分比误差: %.6f' % mape1)
    return rmse, mae, mape1


if __name__ == '__main__':
    # 数据初始化
    look_head = 3
    look_back = look_head
    # 数据导入
    filename = r'D:\LSTM-4.0-processing\AbileneFlow.csv'
    data = load_data(filename)
    # 创建数据集
    data_x, data_y = create_datasets(data, look_head, look_back)
    data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1]))
    data_y = np.reshape(data_y, (data_y.shape[0], data_y.shape[1]))
    df_x = pd.DataFrame(data_x)
    df_x.to_csv(r'D:\LSTM-4.0-processing\data\data_x.csv')
    df_y = pd.DataFrame(data_y)
    df_y.to_csv(r'D:\LSTM-4.0-processing\data\data_y.csv')


