# %%


from matplotlib import pyplot as plt
import numpy as np

def plot_prediction(y_test,y_predicted):
    imgDim=28
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(y_test,[imgDim,imgDim]))
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(y_predicted,[imgDim,imgDim]))
    
    plt.show
