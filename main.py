# %%

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pdb
import generate_dataset as gd
import data_plot as dp


x_train, y_train,x_test, y_test=gd.create_dataset()

# %%
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

def build_RNN_model():
     
    rnn_layer = keras.layers.RNN(
        layers.SimpleRNNCell(units), input_shape=(None, input_dim)
    )
    model = keras.models.Sequential(
        [
            rnn_layer,
            keras.layers.Dense(output_dim),
        ]
    )
    return model





batch_size = 100

model = build_LSTM_model()

model.compile(optimizer='adam', loss='mse',
              metrics=[tf.keras.metrics.MeanSquaredError()])

model.fit(x_train, y_train, validation_data=(x_test, y_test),
          batch_size=batch_size, epochs=10)


y_predicted = model.predict(x_test)
test_sample=0
dp.plot_prediction(y_test[test_sample],y_predicted[test_sample])

    