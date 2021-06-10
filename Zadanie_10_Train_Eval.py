from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import numpy as np
import pandas as pd

x=pd.read_csv('10_x.csv')
y=pd.read_csv('10_y.csv')

x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.2,train_size=0.8, random_state=21)

NeuralModel = Sequential([
                        Dense(128, activation='relu', input_shape=(14,)),
                        Dense(32, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(1, activation='sigmoid')
])


#https://keras.io/api/losses/
#https://keras.io/api/optimizers/
#https://keras.io/api/metrics/

opt = Adam(lr=0.0003)
NeuralModel.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy','AUC'])
NeuralModel.fit(x_train, y_train, batch_size= 16, epochs = 16) #verbose = 1

y_pred = NeuralModel.predict(x_test)
y_pred = np.around(y_pred, decimals=0)
results = accuracy_score(y_test,y_pred)


text_file = open("sample.txt", "w")
n = text_file.write(f"accuracy: {results}")
text_file.close()

print(f"accuracy: {results}")

# Accuracy wynosi 1 z powodu banalnego podziału na 2 klasy jakosci Wina: "bad" i "nice".