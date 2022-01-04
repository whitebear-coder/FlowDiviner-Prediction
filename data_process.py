import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd

filename = r'/home/hp/LZX/GMHCN/data/GeantFlow.xls'
data = pd.read_excel(filename).iloc[:, 0]
data = np.array(data)
data[2219] = (data[2218] + data[2220]) / 2
k = (data[len(data)-1] - data[0]) / len(data)
for i in range(0, len(data)):
    if data[i] == 0:
        data[i] = data[0] + k * i
    elif data[i] > 1e5:
        data[i] = data[0] + k * i

plt.plot(data)plt.show()

data = pd.DataFrame(data)
data.to_csv(r'/home/hp/LZX/GMHCN/data/GeantFlow.csv')
