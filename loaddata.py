import numpy as np

from tensorflow.keras.datasets.mnist import load_data as loaddata

#def loaddata():
#    f=np.load("/work/msimklue/pap/data.npz")
#    return (f["train_x"],f["train_y"]),(f["test_x"],f["test_y"])

if __name__=="__main__":
    (x,y),(tx,ty)=loaddata()

    print(x.shape)
