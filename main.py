# %%

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pdb
import generate_dataset as gd
import data_plot as dp

input_dim=28*28
output_dim=28*28
units=128
def build_LSTM_model():
     
    lstm_layer = keras.layers.RNN(
        keras.layers.LSTMCell(units), input_shape=(None, input_dim)
    )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.Dense(output_dim),
        ]
    )
    return model



x_train, y_train,x_test, y_test=gd.create_dataset()

# %%
batch_size = 100

model = build_LSTM_model()

model.compile(optimizer='sgd', loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])

model.fit(x_train, y_train, validation_data=(x_test, y_test),
          batch_size=batch_size, epochs=50)


# %%
y_predicted = model.predict(x_test)
test_sample=0
dp.plot_prediction(y_test[test_sample],y_predicted[test_sample])