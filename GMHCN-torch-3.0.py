import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from dgl.nn.pytorch import GATConv, GraphConv
from torch_geometric.nn import GCNConv
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
lr = 0.01
epoch = 1
num_heads = 6  # Multi_heads的个数
num_block = 3
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
    return datasets[int(train_size * i):int(train_size) * (i+1)]


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
    def __init__(self, in_feat, out_feat, heads, num_block):
        super(GMHCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.heads = heads
        self.num_blocks = num_block

        self.GCA1 = GCA(self.in_feat, 12, self.heads)
        # self.gcn1 = GraphConv(self.in_feat, 12)
        # self.gat1 = GATConv(12, 24, self.heads)

        # self.GCA = GCA(12 * num_heads)
        self.gcn2 = GraphConv(12 * num_heads, 128)
        self.gat2 = GATConv(128, 256, self.heads)

        self.gcn3 = GraphConv(256 * num_heads, 512)
        self.gat3 = GATConv(512, 1600, self.heads)

        self.encoder = nn.Sequential(
            nn.Conv2d(96, 64, (3, 3)),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(64, 32, (3, 3)),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        # Middle Attention
        self.ma = GATConv(32, 16, self.heads)

        # Decoder
        self.gcn4 = GraphConv(16 * num_heads, 100)
        self.gat4 = GATConv(100, 50, self.heads)

        self.gcn5 = GraphConv(50 * num_heads, 200)
        self.gat5 = GATConv(200, 100, self.heads)

        self.gcn6 = GraphConv(100 * num_heads, 400)
        self.gat6 = GATConv(400, 800, num_heads)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, (3, 3)),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(24, 12, (3, 3)),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 3, (3, 3)),  # b, 1, 28, 28
            nn.Tanh()
        )

        self.dense = nn.Linear(768, 36)

    def forward(self, g, features, training=True, mask=None):
        self.g = g
        x = self.GCA1(self.g, features)
        # x = self.gcn1(g, features)
        # x = self.gat1(self.g, x)
        print(x.shape)

        x = torch.reshape(x, (x.shape[0], -1))


        x = self.gcn2(self.g, x)
        x = self.gat2(self.g, x)
        x = torch.reshape(x, (x.shape[0], -1))

        x = self.gcn3(self.g, x)
        x = self.gat3(self.g, x)

        # x = torch.reshape(x, (x.shape[0], -1, 1, 1))
        x = torch.reshape(x, (x.shape[0], -1, 10, 10))
        x = self.encoder(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.ma(self.g, x)

        x = torch.reshape(x, (x.shape[0], -1))

        x = self.gcn4(self.g, x)
        x = self.gat4(self.g, x)
        x = torch.reshape(x, (x.shape[0], -1))

        x = self.gcn5(self.g, x)
        x = self.gat5(self.g, x)
        x = torch.reshape(x, (x.shape[0], -1))

        x = self.gcn6(self.g, x)
        x = self.gat6(self.g, x)
        x = torch.reshape(x, (x.shape[0], -1, 10, 10))

        x = self.decoder(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.dense(x)
        return x


if __name__ == '__main__':
    # 数据初始化
    look_head = 3
    look_back = look_head
    # 数据导入
    filename = r'D:\LSTM-4.0-processing\AbileneFlow.csv'
    data = pd.read_csv(filename)
    data = np.array(data)
    data = data[:100]
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
    # 图的建立
    g = dgl.graph(([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 8, 9, 9, 10, 12],
                    [1, 2, 3, 2, 3, 9, 10, 3, 4, 11, 13, 5, 11, 13, 7, 8, 9, 11, 10, 12, 14, 14, 10, 11, 12, 14]))
    g = dgl.add_self_loop(g)

    A_betweenness_value = np.array(
        [0.1666666669, 0.2803030303, 0.1616161616, 0.2045454545, 0.2045454545, 0.09848484848, 0.2727272727, 0.08383838383,
         0.1136363636, 0.1439393939, 0.2121212121, 0.2651515151, 0.13636363636, 0.9848484848, 0.1136363636])

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
    model = GMHCN(36, 36, num_heads, num_block)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
    criterion = nn.MSELoss()

    # 初始化图
    for epoch_child in range(10):
        for i in range(int(x_label_array.shape[1]/(batch_size * 3))):
            feature = x_data_array[:, int(i * batch_size * 3):int((i + 1) * batch_size * 3)]
            labels = x_label_array[:, int(i * batch_size * 3):int((i + 1) * batch_size * 3)]
            logits = model(g, feature)
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
            print("% d / 3206" % i)
            print(feature.shape[1])

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
        labels = y_label_array[:, int(i * batch_size * 3):int((i + 1) * batch_size * 3)]
        logits = model(g, feature)
        loss = criterion(logits, labels)
        print(i)
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
        print(i)

    features = sc.inverse_transform(features)
    labelss = sc.inverse_transform(labelss)
    features = np.expm1(features)
    labelss = np.expm1(labelss)

    test_logits = sc.inverse_transform(features)
    test_labels = sc.inverse_transform(labelss)
    evaluate_function(test_logits, test_labels)





