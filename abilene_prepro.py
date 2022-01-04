import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filename = r'/home/hp/LZX/GMHCN/data/AbileneFlow.xls'
data = pd.read_excel(filename)
data = np.array(data)
'''
k = (data[4000] - data[0]) / len(data)
for i in range(0, 4000):
    if data[i] == 0:
        data[i] = data[0] + k * i
    elif data[i] > 2.5e8:
        data[i] = data[0] + k * i

for i in range(4000, data.shape[0]-1):
    if data[i] == 0:
        data[i] = data[0] + k * i
    elif data[i] > 2.5e8:
        data[i] = data[0] + k * i
'''
plt.plot(data)
plt.show()

'''
data = pd.DataFrame(data)
data.to_csv(r'/home/hp/LZX/GMHCN/data/BrainFlow.csv')
'''