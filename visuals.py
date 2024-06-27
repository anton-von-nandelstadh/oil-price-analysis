import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

data = pd.read_csv('data/oilprice.csv')
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.set_index('DATE')
data['DCOILBRENTEU'].replace('.', np.NaN, inplace=True)
data = data.ffill()
data['DCOILBRENTEU'] = data['DCOILBRENTEU'].astype('float32')

plt.plot(data.index, data['DCOILBRENTEU'])
plt.show()
