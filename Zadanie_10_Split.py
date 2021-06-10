from sklearn.preprocessing import StandardScaler, LabelEncoder

import numpy as np
import pandas as pd

wine=pd.read_csv('winequality-red.csv')

y = wine['quality']
x = wine.drop('quality', axis=1)

citricacid = x['fixed acidity'] * x['citric acid']
citric_acidity = pd.DataFrame(citricacid, columns=['citric_accidity'])
density_acidity = x['fixed acidity'] * x['density']
density_acidity = pd.DataFrame(density_acidity, columns=['density_acidity'])

x = wine.join(citric_acidity).join(density_acidity)

bins = (2, 5, 8)
labels = ['bad', 'nice']
y = pd.cut(y, bins = bins, labels = labels)

enc = LabelEncoder()
yenc = enc.fit_transform(y)

scale = StandardScaler()
scaled_x = scale.fit_transform(x)

df_x = pd.DataFrame(scaled_x)
df_y = pd.DataFrame(yenc)

df_x.to_csv(r'10_x.csv', index=False)
df_y.to_csv(r'10_y.csv', index=False)