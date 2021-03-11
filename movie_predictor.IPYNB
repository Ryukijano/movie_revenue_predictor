import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

data_x = pd.read_csv('C:/Users/Ryukijano/Movie_collection_target.csv', header =0)
data_y = pd.read_csv('C:/Users/Ryukijano/Movie_collection_independent.csv', header =0)
data_x.iloc[0:5]
data_x.describe
data_y.iloc[0:5]
data_y.describe

from sklearn.model_selection import train_test_split
X_train_full,X_test,y_train_full, y_test = train_test_split(data_y,data_x, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

np.random.seed(42)
tf.random.set_seed(42)

X_train.shape

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[19]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])

model.summary()

model.compile(loss="mean_squared_error",
             optimizer=keras.optimizers.SGD(lr=1e-3),
             metrics=['mae'])
						 
						 
model_history = model.fit(X_train , y_train , epochs=100, validation_data=(X_valid, y_valid))
mae_test = model.evaluate(X_test, y_test)
model_history.history


pd.DataFrame(model_history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

X_new = X_test[:5]

y_pred = model.predict(X_new)
print(y_pred)
print(y_test[:5])

y_train.shape

X_test.shape
