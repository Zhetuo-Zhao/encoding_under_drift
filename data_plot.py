# %%


from matplotlib import pyplot as plt
import numpy as np
import pdb

def plot_prediction(a,b):
    imgDim=28
    plt.subplot(1,2,1)
    fig=plt.imshow(np.reshape(a,[imgDim,imgDim]))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
    plt.subplot(1,2,2)
    fig=plt.imshow(np.reshape(b,[imgDim,imgDim]))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
    plt.show
    

def plot_retinaFlow(x_test,sampleIdx):
    #pdb.set_trace()
    imgDim=28
    for i in range(x_test.shape[1]):
        plt.imshow(np.reshape(x_test[sampleIdx,i,],[imgDim,imgDim]))
        plt.show()
        #pdb.set_trace()
        #plt.savefig('retinaFlow'+str(i)+'.png')
        
        
        

    
