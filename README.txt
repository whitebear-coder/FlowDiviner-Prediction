 数据导入
数据来源网址 http://sndlib.zib.de/home.action
AbileneFlow
BrainFlow
GeantFlow
# 数据
边的邻接矩阵
边的betweenness
总流量
# 数据预处理
1.MINMAX_scaler
2.lop1p平滑
3.小波变换去噪声
# 模型部分
1.图神经网络＋图的注意力机制
2.encoder + 中间 + decoder
3.跳联结构 ---->参照ResNet
# 后处理部分
1.MinMax_scaler还原
2.exp1p反平滑
# 评价指标
rmse
mse
mape

