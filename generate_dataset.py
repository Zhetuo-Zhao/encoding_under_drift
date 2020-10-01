# %%
from tensorflow import keras
from numpy as np
mnist = keras.datasets.mnist

(x_train0, y_train), (x_test0, y_test) = mnist.load_data()
x_train0, x_test0 = x_train0 / 255.0, x_test0 / 255.0

imgDim=28
time_steps=20

for i in range(x_train0.shape[0]):
    img0=x_train0[i]
    eyeX, eyeY=0,0
    displaceX, displaceY = np.random.normal(0, 1, time_steps),  np.random.normal(0, 1, time_steps)
    for t in range(time_steps):
        frameT=np.zeros((imgDim,imgDim))
        img0(max())
        frameT
        frameT+=np.random.normal(0, 0.5, (imgDim,imgDim))
        eyeX+=displaceX[t]
        eyeY+=displaceY[t]
        
    