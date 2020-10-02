# %%
from tensorflow import keras
import numpy as np
import pdb
from matplotlib import pyplot as plt
import progressbar
from tqdm import tqdm

def generate_retinal_flow(Data,time_steps,D,NoiseRate):
    print('generating retinal flow')
    samples=Data.shape[0]
    imgDim=Data.shape[1];
    retinal_flow=np.zeros((samples, time_steps, imgDim*imgDim))
    label=np.zeros((samples, imgDim*imgDim))
    logStep=500
    
    #bar = progressbar.ProgressBar(maxval=samples, \
    #widgets=[progressbar.Bar('=', '[', ']'), '\n', progressbar.Percentage()])
    #bar.start()
    for i in tqdm(range(samples)):
        img0=Data[i]
        label[i,]=img0.flatten()
        
        eyeX, eyeY=0.0,0.0
        displaceX, displaceY = np.random.normal(0, D, time_steps),  np.random.normal(0, 1, time_steps)
        for t in range(time_steps):
            frameT=np.zeros((imgDim,imgDim))
            
            if abs(eyeX)>24 or abs(eyeY)>24:
                eyeX, eyeY= 0,0
                
            imgT=img0[max(round(eyeX),0):min(round(eyeX)+imgDim,imgDim+1),
                      max(round(eyeY),0):min(round(eyeY)+imgDim,imgDim+1)]
                
            frameT[max(round(-eyeX),0):min(round(-eyeX)+imgDim,imgDim+1),
                      max(round(-eyeY),0):min(round(-eyeY)+imgDim,imgDim+1)]=imgT
            frameT+=np.random.normal(0, NoiseRate, (imgDim,imgDim))
            
            retinal_flow[i,t,]=frameT.flatten()
            eyeX+=displaceX[t]
            eyeY+=displaceY[t]
    
            #bar.update(i+1)
            #plt.imshow(frameT)
            #plt.show()
            #pdb.set_trace()
    #bar.finish()      
            
    return retinal_flow, label
        
def create_dataset():
    mnist = keras.datasets.mnist
    
    (x_train0, y_train0), (x_test0, y_test0) = mnist.load_data()
    x_train0, x_test0 = x_train0 / 255.0, x_test0 / 255.0
    
    time_steps=20
    diffussion=1.5
    NoiseRate=0.5
    x_train,y_train=generate_retinal_flow(x_train0,time_steps,diffussion,NoiseRate)
    x_test,y_test=generate_retinal_flow(x_test0,time_steps,diffussion,NoiseRate)
    
    return x_train, y_train, x_test,y_test




 