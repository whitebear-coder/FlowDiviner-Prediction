from xml.dom import minidom
import os
import networkx
import numpy as np
import pandas as pd
import pywt
import torch
from sklearn.preprocessing import MinMaxScaler
import xlrd

# 初始化
graph = networkx.Graph()
betweeness_graph = networkx.Graph()
node_index = 0
nodes_dict = {}
nodes_locate_list = []


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


# 引入数据
def preprocessing(data):
    sc = MinMaxScaler(feature_range=(0, 1))
    data = sc.fit_transform(data)

    data = np.log1p(data)

    data = data.flatten()
    series = []
    for i in range(len(data)):
        series.append(data[i])  # 将数据转化为int类型
    series = wavelet_denoise(series)

    return series


# total_flow = 4000
total_flow = np.array(pd.read_excel('D:\LSTM-4.0-processing\AbileneFlow.xls'))
# print(total_flow.shape)
total_flow = preprocessing(total_flow)
# 导入包的位置

node_path = r'D:\LSTM-4.0-processing\data\all_total_data.xml'
doc = minidom.parse(node_path)
# 数据数目
data_num = 500
# 取每个点的位置
nodes = doc.getElementsByTagName("node")

for node in nodes:
    sid = node.getAttribute("id")
    x = node.getElementsByTagName("x")[0]
    y = node.getElementsByTagName("y")[0]
    d = {node_index: sid}
    # node index 和 name 加入字典
    nodes_dict.update(d)
    # node locate --> nodes_locate_list
    nodes_locate_list.append((float(x.firstChild.data), float(y.firstChild.data)))
    # 节点加入图结构中
    graph.add_node(sid)
    betweeness_graph.add_node(sid)
    # 更新
    node_index += 1

# 求边的betweenness
links = doc.getElementsByTagName("link")
for link in links:
    source = link.getElementsByTagName("source")[0]
    target = link.getElementsByTagName("target")[0]
    capacity = link.getElementsByTagName("capacity")[0]
    graph.add_edge(source.firstChild.data, target.firstChild.data, weight=capacity.firstChild.data)


# 点节点度和边节点度
node_betweenness = networkx.betweenness_centrality(graph)
edge_betweenness = networkx.edge_betweenness_centrality(graph)


# 求节点度流量矩阵
edge_betweenness_index_list = list(edge_betweenness.keys())
edge_betweenness_value_list = list(edge_betweenness.values())
edge_betweenness_list = []
for i in range(len(edge_betweenness_value_list)):
    betweeness_graph.add_edge(edge_betweenness_index_list[i][0], edge_betweenness_index_list[i][1], weight=float(edge_betweenness_value_list[i]))

# 求邻接矩阵
adjacency_matrix_none = np.array(networkx.adjacency_matrix(betweeness_graph, weight=None).todense()) + np.eye(12, 12)
diagonal_matrix = np.eye(12, 12)
node_degree = np.power(np.array([1, 4, 2, 3, 3, 3, 3, 2, 2, 3, 2, 2]), -0.5).reshape(-1, 1)
degree_matrix = node_degree * diagonal_matrix

A = degree_matrix @ adjacency_matrix_none @ degree_matrix
df = pd.DataFrame(A)
df.to_csv(r'D:\LSTM-4.0-processing\\' + 'A.txt')

# 求节点度矩阵
# 1, 4, 2, 3, 3, 3, 3, 2, 2, 3, 2, 2



