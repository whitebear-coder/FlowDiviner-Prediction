import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from dgl.nn.pytorch import GATConv, GraphConv
# from torch_geometric.nn import GCNConv
import dgl.function as fn
import dgl
from sklearn.preprocessing import MinMaxScaler
import math
import sklearn.metrics as metrics
import pywt
import time
from sklearn.model_selection import train_test_split
from preprocess import rev_trans
from dgl.data.citation_graph import CoraGraphDataset

torch.set_default_tensor_type(torch.DoubleTensor)

# 设置超参
lr = 0.0001
epoch = 2000
num_heads = 6  # Multi_heads的个数
num_block_Q = 6
num_block_L = 6
batch_size = 12


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
    return datasets[int(train_size * i):int(train_size) * (i + 1)]


def evaluate(pred_set_end, labels_set_end):
    mape1 = mape(pred_set_end, labels_set_end)
    rmse = math.sqrt(metrics.mean_squared_error(pred_set_end, labels_set_end))
    mae = metrics.mean_absolute_error(pred_set_end, labels_set_end)

    print('均方根误差: %.6f' % rmse)
    print('平均绝对误差: %.6f' % mae)
    print('平均百分比误差: %.6f' % mape1)
    return rmse, mae, mape1


def evaluate_function(pred_set_end, labels_set_end):
    mape1 = mape(pred_set_end, labels_set_end)
    rmse = math.sqrt(metrics.mean_squared_error(pred_set_end, labels_set_end))
    mae = metrics.mean_absolute_error(pred_set_end, labels_set_end)

    print('均方根误差: %.6f' % rmse)
    print('平均绝对误差: %.6f' % mae)
    print('平均百分比误差: %.6f' % mape1)
    # return rmse, mae, mape1


class GCA(nn.Module):
    def __init__(self, in_feat, out_feat, heads):
        super(GCA, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.heads = heads

        self.gcn1 = GraphConv(self.in_feat, 36)
        self.gat1 = GATConv(36, out_feat, self.heads)

    def forward(self, g, features):
        self.g = g
        x = self.gcn1(self.g, features)
        x = self.gat1(self.g, x)
        return x


class GMHCN(nn.Module):
    def __init__(self, in_feat, out_feat, heads):
        super(GMHCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.heads = heads

        self.GCA1 = GCA(self.in_feat, 6, self.heads)

        self.GCA = GCA(6 * num_heads, 6, self.heads)

        self.ma = GATConv(36, 6, self.heads)

        self.dense = nn.Linear(36, 36)

    def forward(self, g, features, num_blocks_Q, num_blocks_L, training=True, mask=None):
        self.g = g
        x = self.GCA1(self.g, features)
        x = torch.reshape(x, (x.shape[0], -1))
        # print(x.shape)
        x1 = x
        # encoder
        for _ in range(int(num_blocks_Q/2)):
            indentify = x
            x = self.GCA(self.g, x)
            x = torch.reshape(x, (x.shape[0], -1))
            # 跳联
            # print(x.shape)
            x = self.GCA(self.g, x)
            x = torch.reshape(x, (x.shape[0], -1))
            # print(x.shape)
            x = indentify + x
        # middle
        # print(x.shape, features.shape)
        x = self.ma(self.g, x)
        x = torch.reshape(x, (x.shape[0], -1))
        # decoder
        # print(x.shape)
        x = self.GCA1(self.g, x)
        x = torch.reshape(x, (x.shape[0], -1))
        for _ in range(int(num_blocks_L/2)):
            indentify = x
            x = self.GCA(self.g, x)
            x = torch.reshape(x, (x.shape[0], -1))
            # 跳联
            # print(x.shape)
            x = self.GCA(self.g, x)
            x = torch.reshape(x, (x.shape[0], -1))
            # print(x.shape)
            x = indentify + x
        x = self.dense(x)
        return x


if __name__ == '__main__':
    # 数据初始化
    look_head = 3
    look_back = look_head
    # 数据导入
    filename_1 = r'/home/hp/LZX/GMHCN/data/AbileneFlow_yuan.xls'
    filename_2 = r'/home/hp/LZX/GMHCN/data/BrainFlow_yuan.xls'
    filename_3 = r'/home/hp/LZX/GMHCN/data/GeantFlow_yuan.xls'
    filename = filename_2
    data = pd.read_excel(filename).iloc[:, 0]
    data = np.array(data)
    # data = data[data < 200000]
    # data = data[:1000]
    data = data.ravel()
    data = wavelet_denoise(data)
    data = data.reshape(-1, 1)
    data = smooth(data)
    sc = MinMaxScaler(feature_range=(0, 1))
    data = sc.fit_transform(data)
    series = []
    for i in range(len(data)):
        series.append(list(map(float, data[i])))  # 将数据转化为int类型

    # 创建数据集
    data, label = create_datasets(data, look_head, look_back)
    data = np.reshape(data, (data.shape[0], data.shape[1]))
    label = np.reshape(label, (label.shape[0], label.shape[1]))
    # 训练集和测试集分离
    x_data, y_data, x_label, y_label = train_test_split(data, label, train_size=0.8)
    if filename == filename_1:
        # 图的建立
        g = dgl.graph(([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 8, 9, 9, 10, 12],
                       [1, 2, 3, 2, 3, 9, 10, 3, 4, 11, 13, 5, 11, 13, 7, 8, 9, 11, 10, 12, 14, 14, 10, 11, 12, 14]))
        g = dgl.add_self_loop(g)

        A_betweenness_value = np.array(
            [0.1666666669, 0.2803030303, 0.1616161616, 0.2045454545, 0.2045454545, 0.09848484848, 0.2727272727,
             0.08383838383,
             0.1136363636, 0.1439393939, 0.2121212121, 0.2651515151, 0.13636363636, 0.9848484848, 0.1136363636])
    elif filename == filename_2:
        
    # train_feature = np.array(pd.read_csv(r'D:\LSTM-4.0-processing\data\data_x.csv'))
    # train_feature = sc.fit_transform(train_feature[:, 1:4])
    # test_feature = np.array(pd.read_csv(r'D:\LSTM-4.0-processing\data\data_x.csv'))
    # test_feature = sc.fit_transform(test_feature[:, 1:4])
    for i in range(A_betweenness_value.shape[0]):
        if i == 0:
            x_data_array = A_betweenness_value[0] * x_data
            x_label_array = A_betweenness_value[0] * x_label
        else:
            x_data_array = np.concatenate((x_data_array, A_betweenness_value[0] * x_data), axis=1)
            x_label_array = np.concatenate((x_label_array, A_betweenness_value[0] * x_label), axis=1)

    x_data_array = np.reshape(x_data_array, (15, -1))
    x_label_array = np.reshape(x_label_array, (15, -1))

    x_data_array = torch.tensor(x_data_array, requires_grad=True)
    x_label_array = torch.tensor(x_label_array, requires_grad=True)
    model = GMHCN(36, 36, num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
    criterion = nn.MSELoss()

    # 初始化图
    for epoch_child in range(epoch):
        for i in range(int(x_label_array.shape[1] / (batch_size * 3))):
            feature = x_data_array[:, int(i * batch_size * 3):int((i + 1) * batch_size * 3)]
            labels = x_label_array[:, int(i * batch_size * 3):int((i + 1) * batch_size * 3)]
            logits = model(g, feature, num_block_Q, num_block_L)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0:
                feature = logits.detach().numpy()
                labels = labels.detach().numpy()
                features = feature
                labelss = labels
            else:
                feature = logits.detach().numpy()
                labels = labels.detach().numpy()
                features = np.concatenate((features, feature), axis=1)
                labelss = np.concatenate((labelss, labels), axis=1)
            if i % 20 == 0:
                print("第{0}轮:batch:{1}/{2}".format(epoch_child, i, int(x_label_array.shape[1] / (batch_size * 3))))

    # ------------------------测试集部分--------------------------
    for i in range(A_betweenness_value.shape[0]):
        if i == 0:
            y_data_array = A_betweenness_value[0] * y_data
            y_label_array = A_betweenness_value[0] * y_label
        else:
            y_data_array = np.concatenate((y_data_array, A_betweenness_value[0] * y_data), axis=1)
            y_label_array = np.concatenate((y_label_array, A_betweenness_value[0] * y_label), axis=1)

    y_data_array = np.reshape(y_data_array, (15, -1))
    y_label_array = np.reshape(y_label_array, (15, -1))

    y_data_array = torch.tensor(y_data_array, requires_grad=True)
    y_label_array = torch.tensor(y_label_array, requires_grad=True)

    for i in range(int(y_label_array.shape[1] / (batch_size * 3))):
        feature = y_data_array[:, int(i * batch_size * 3):int((i + 1) * batch_size * 3)]
        print(feature)
        labels = y_label_array[:, int(i * batch_size * 3):int((i + 1) * batch_size * 3)]
        logits = model(g, feature, num_block_Q, num_block_L)
        print(logits)
        loss = criterion(logits, labels)

        if i == 0:
            feature = logits.detach().numpy()
            labels = labels.detach().numpy()
            features = feature
            labelss = labels
        else:
            feature = logits.detach().numpy()
            labels = labels.detach().numpy()
            features = np.concatenate((features, feature), axis=1)
            labelss = np.concatenate((labelss, labels), axis=1)


    features = sc.inverse_transform(features)
    labelss = sc.inverse_transform(labelss)
    features = np.expm1(features)
    labelss = np.expm1(labelss)

    test_logits = sc.inverse_transform(features)
    test_labels = sc.inverse_transform(labelss)
    evaluate_function(test_logits, test_labels)





