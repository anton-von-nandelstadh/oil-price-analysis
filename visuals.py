import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('data/oilprice.csv')
print(data.index)
print(data.head)
data = data.set_index('DATE')
data['DCOILBRENTEU'].replace('.', np.NaN, inplace=True)
print(data.head)
data = data.ffill()
data['DCOILBRENTEU'].astype('float')

plt.plot(data[['DCOILBRENTEU']].values.astype('float32'))
plt.show()
