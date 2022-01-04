# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from dgl.nn.pytorch import GATConv, GraphConv
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
from prepreprocess import read_graph
from torch.utils.data import DataLoader
import logger

torch.set_default_tensor_type(torch.DoubleTensor)

# 设置超参
lr = 0.00001
# epoch0=0代表从头开始训练，epoch0=800等于断点续训
epoch0 = 0
epoch = 1000
# 数据初始化
look_head = 12
look_back = 3
num_heads = 6  # Multi_heads的个数
num_block_Q = 6
num_block_L = 6
batch_size = 64
Q = 3
filename = r'/home/hp/LZX/GMHCN/data/BrainFlow.xls'
g_name = 'BrainFlow'
model_param_save = '/home/hp/LZX/GMHCN/checkpoints'
model_pred = '/home/hp/LZX/GMHCN/model_pred'
model_total_pred = '/home/hp/LZX/GMHCN/model_total_pred'


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


def create_datasets(datasets, look_head, look_back):
    data_x = []
    data_y = []
    for i in range(len(datasets) - look_head - look_head + 1):
        window = datasets[i:(i + look_head)]
        data_x.append(window)
        data_y.append(datasets[i + look_head:i + look_head + look_back])
    return np.array(data_x), np.array(data_y)


def mape(y_pred, y_true):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def split_k(datasets, k, i):
    train_size = int(len(datasets) * (1 / k))
    return datasets[int(train_size * i):int(train_size) * (i + 1)]


def evaluate_function(pred_set_end, labels_set_end):
    mape1 = mape(pred_set_end, labels_set_end)
    rmse = math.sqrt(metrics.mean_squared_error(pred_set_end, labels_set_end))
    mae = metrics.mean_absolute_error(pred_set_end, labels_set_end)
    print("epoch:{0}, lr:{1}, look_head:{2}， look_back:{3}".format(epoch, lr, look_head, look_back))
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

        self.ma = GATConv(36, int(look_head * (batch_size / 6)), self.heads)

        self.dense = nn.Linear(36, (look_back * batch_size))

    def forward(self, g, features, num_blocks_Q, num_blocks_L, training=True, mask=None):
        self.g = g

        x = self.GCA1(self.g, features)

        x = torch.reshape(x, (x.shape[0], -1))
        x1 = x
        # encoder
        for _ in range(int(num_blocks_Q / 2)):
            indentify = x
            x = self.GCA(self.g, x)
            x = torch.reshape(x, (x.shape[0], -1))
            x = self.GCA(self.g, x)
            x = torch.reshape(x, (x.shape[0], -1))
            x = indentify + x
            # nn.Dropout(0.2)
        # middle
        x = self.ma(self.g, x)
        x = torch.reshape(x, (x.shape[0], -1))
        # decoder
        x = self.GCA1(self.g, x)
        x = torch.reshape(x, (x.shape[0], -1))
        # nn.Dropout(0.2)
        for _ in range(int(num_blocks_L / 2)):
            indentify = x
            x = self.GCA(self.g, x)
            x = torch.reshape(x, (x.shape[0], -1))
            # 跳联
            x = self.GCA(self.g, x)
            x = torch.reshape(x, (x.shape[0], -1))
            x = indentify + x
            # nn.Dropout(0.2)
        x = self.dense(x)
        return x


def save_checkpoint_state(epoch, model, optimizer, model_param_save, run_name):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if not os.path.isdir(model_param_save):
        os.mkdir(model_param_save)

    torch.save(checkpoint, os.path.join(model_param_save, run_name + '.pth'))


def get_checkpoint_state(dir, model, optimizer, run_name):
    # 恢复上次的训练状态
    # logger.info("Resume from checkpoint...")
    print("Resume from checkpoint...")
    checkpoint = torch.load(os.path.join(dir, run_name + '.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # logger.info('sucessfully recover from the last state')
    print('sucessfully recover from the last state')
    return model, epoch, optimizer


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # ---------------数据导入---------------
    data = pd.read_excel(filename)
    data = np.array(data)
    # data_ori 原数据
    data_ori = data
    # 小波变换
    data = data.ravel()
    data = wavelet_denoise(data)
    # data_ori2 小波变换后的数据
    data = data.reshape(-1, 1)
    data_ori2 = data
    # 数据归一化
    sc = MinMaxScaler(feature_range=(0, 1))
    # ---------------------------------------
    # g->图网络结构， A_betweenness_value->节点度，nodes->节点数目
    g, A_betweenness_value, nodes = read_graph(g_name)
    A_betweenness_value = np.array(A_betweenness_value)
    # 每个节点的节点度乘x_data, x_label转换成每个节点对应的信息
    for i in range(A_betweenness_value.shape[0]):
        if i == 0:
            data_sets = A_betweenness_value[i] * data
        else:
            data_w = A_betweenness_value[i] * data
            data_sets = np.concatenate((data_sets, data_w), axis=1)

    # 训练集和测试集分离
    train_set, test_set = train_test_split(data_sets, train_size=0.8, shuffle=False)

    train_set = np.log1p(train_set)
    sc = MinMaxScaler(feature_range=(0, 1))
    train_set = sc.fit_transform(train_set)

    train_data, train_label = create_datasets(train_set, look_head, look_back)

    train_data = train_data.reshape(-1, nodes)
    train_label = train_label.reshape(-1, nodes)

    train_data = np.transpose(train_data)
    train_label = np.transpose(train_label)

    x_data_array = torch.tensor(train_data, requires_grad=True)
    x_label_array = torch.tensor(train_label, requires_grad=True)

    # -----------------------模型初始化部分------------------------
    # 模型加优化器的设定
    model = GMHCN(look_head * batch_size, look_head * batch_size, num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    criterion = nn.MSELoss()

    run_name = 'GMHCN_train' + '_' + str(g_name) + '_' + 'lr' + str(lr) + '_' + 'epoch' + str(epoch0) + '_bs' + str(
        batch_size) + '_' + 'lh' + str(look_head) + '_' + 'lb' + str(look_back) + '_' + 'checkpoints'

    if epoch0 != 0:
        model, epoch0, optimizer = get_checkpoint_state(model_param_save, model, optimizer, run_name)
    losses = np.array([])

    epoch_end = epoch-1
    # -----------------------模型训练部分--------------------------
    for epoch_child in range(epoch0, epoch):
        # 分批放入x_label_array和x_data_array
        for i in range(int(x_label_array.shape[1] / (batch_size * look_back))):
            # 特征shape->(22, 16*12(batch_size*look_head))
            feature = x_data_array[:, int(i * batch_size * look_head):int((i + 1) * batch_size * look_head)]
            # 特征shape->(22, 16*3(batch_size*look_back))
            labels = x_label_array[:, int(i * batch_size * look_back):int((i + 1) * batch_size * look_back)]
            # logits输出->(22, 16*3(batch_size*look_back))
            logits = model(g, feature, num_block_Q, num_block_L)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch:{0}:loss:{1}".format(epoch_child, loss.item()))
        losses = np.append(losses, loss.item())
        '''
        if loss.item() < 0.001:
            run_name = 'GMHCN_train' + '_' + str(g_name) + '_' + 'lr' + str(lr) + '_' + 'epoch' + str(epoch_child) + '_bs' + str(batch_size) + '_' + 'lh' + str(look_head) + '_' + 'lb' + str(look_back) + '_' + 'checkpoints'
            epoch_end = epoch_child
            break
        # print('---------logits-----------')
        # print(logits)
        # print('---------labels-----------')
        # print(labels)
        '''
    # ------------------------模型保存---------------------------
    if epoch_end == epoch-1:
        run_name = 'GMHCN_train' + '_' + str(g_name) + '_' + 'lr' + str(lr) + '_' + 'epoch' + str(epoch) + '_bs' + str(
            batch_size) + '_' + 'lh' + str(look_head) + '_' + 'lb' + str(look_back) + '_' + 'checkpoints'
    save_checkpoint_state(epoch, model, optimizer, model_param_save, run_name)



    # ------------------------测试集部分--------------------------
    c = test_set
    test_set = np.log1p(test_set)
    sc = MinMaxScaler(feature_range=(0, 1))
    test_set = sc.fit_transform(test_set)

    test_data, test_label = create_datasets(test_set, look_head, look_back)

    test_data = test_data.reshape(-1, nodes)
    test_label = test_label.reshape(-1, nodes)

    test_data = np.transpose(test_data)
    test_label = np.transpose(test_label)

    y_data_array = torch.tensor(test_data, requires_grad=True)
    y_label_array = torch.tensor(test_label, requires_grad=True)

    for i in range(int(y_data_array.shape[1] / (batch_size * look_head))):
        feature = y_data_array[:, int(i * batch_size * look_head):int((i + 1) * batch_size * look_head)]
        label = y_label_array[:, int(i * batch_size * look_back):int((i + 1) * batch_size * look_back)]
        pred = model(g, feature, num_block_Q, num_block_L)
        # 将数据聚合起来
        if i == 0:
            preds = pred.detach().numpy()
            labels = label.detach().numpy()
        else:
            pred = pred.detach().numpy()
            label = label.detach().numpy()
            preds = np.concatenate((preds, pred), axis=1)
            labels = np.concatenate((labels, label), axis=1)
    # -----------------------------模型后处理部分------------------------

    preds = np.transpose(preds)
    labels = np.transpose(labels)

    preds = sc.inverse_transform(preds)
    labels = sc.inverse_transform(labels)

    test_preds = np.expm1(preds)
    test_labels = np.expm1(labels)

    evaluate_function(test_preds.flatten(), test_labels.flatten())
    plt.plot(losses)
    plt.show()
    losses = pd.DataFrame(losses)
    if epoch0 != epoch:
        losses.to_csv(run_name + '.csv')
    # ----------------------存到excel表格中--------------------------------------
    
    features = np.transpose(test_preds)
    labelss = np.transpose(test_labels)
    for i in range(features.shape[1]):
        if i == 0:
            feature_preds = np.array(features[:, i]).reshape(-1, 1)
            feature_labels = np.array(labelss[:, i]).reshape(-1, 1)
        if i % look_back == 0:
            feature_pred = features[:, i].reshape(-1, 1)
            feature_label = labelss[:, i].reshape(-1, 1)
            feature_preds = np.concatenate((feature_preds, feature_pred), axis=1)
            feature_labels = np.concatenate((feature_labels, feature_label), axis=1)

    data = np.concatenate((feature_preds, feature_labels), axis=0)
    data_name = 'GMHCN_test_pred' + '_' + str(g_name) + '_' + str(lr) + '_' + str(epoch) + '_bs' + str(batch_size) + '_' + 'lh' + str(look_head) + '_' + 'lb' + str(look_back)
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(model_pred, data_name + '.csv'))

    data = np.concatenate((feature_preds.sum(axis=0).reshape(-1, 1), feature_labels.sum(axis=0).reshape(-1, 1)), axis=1)
    data_name = 'GMHCN_test_pred_total' + '_' + str(g_name) + '_' + str(lr) + '_' + str(epoch) + '_bs' + str(batch_size) + '_' + 'lh' + str(look_head) + '_' + 'lb' + str(look_back)
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(model_total_pred, data_name + '.csv'))





