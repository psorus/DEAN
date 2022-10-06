import sys
#dex=0

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam,SGD,RMSprop

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import shutil

with open("hyper.json","r") as f:
    hyper=json.loads(f.read())

rounds=hyper["rounds"]#how many models to train
index=hyper["dex"]#which y value should be considered normal
dim=hyper["bag"]#which bagging size to choose for the feature bagging


#load data, and change the shape into (samples, features)
from loaddata import loaddata    
(x_train0, y_train), (x_test0, y_test) = loaddata()
if len(x_train0.shape)>2:
    x_train0=np.reshape(x_train0,(x_train0.shape[0],np.prod(x_train0.shape[1:])))
    x_test0 =np.reshape(x_test0 ,(x_test0.shape[0],np.prod(x_test0.shape[1:])))

#train one model (of index dex)
def train(dex):
    
    pth=f"results/{dex}/"
    
    
    if os.path.isdir(pth):
        shutil.rmtree(pth)
    
    os.makedirs(pth, exist_ok=False)
    
    
    def statinf(q):
        return {"shape":q.shape,"mean":np.mean(q),"std":np.std(q),"min":np.min(q),"max":np.max(q)}
    
    
    x_train, x_test = x_train0.copy(), x_test0.copy()


    #choose some features for the current model
    predim=int(x_train.shape[1])
    to_use=np.random.choice([i for i in range(predim)],dim,replace=False)

    x_train=np.concatenate([np.expand_dims(x_train[:,use],axis=1) for use in to_use],axis=1)
    x_test =np.concatenate([np.expand_dims(x_test [:,use],axis=1) for use in to_use],axis=1)


    #normalise the data, so that the mean is zero, and the standart deviation is one
    norm=np.mean(x_train)
    norm2=np.std(x_train)

    def normalise(q):
        return (q-norm)/norm2
    

    def getdata(x,y,norm=True,normdex=7,n=-1):
        if norm:
            ids=np.where(y==normdex)
        else:
            ids=np.where(y!=normdex)
        qx=x[ids]
        if n>0:qx=qx[:n]
        qy=np.reshape(qx,(int(qx.shape[0]),dim))
        return normalise(qy)
    
    #split data into normal and abnormal samples. Train only on normal ones
    normdex=index
    train=getdata(x_train,y_train,norm=True,normdex=normdex)
    at=getdata(x_test,y_test,norm=False,normdex=normdex)
    t=getdata(x_test,y_test,norm=True,normdex=normdex)
    
   
    #function to build one tensorflow model 
    def getmodel(q,reg=None,act="relu",mean=1.0):
        inn=Input(shape=(dim,))
        w=inn
        for aq in q[1:-1]:
            #change this line to use constant shifts
            #w=Dense(aq,activation=act,use_bias=True,kernel_initializer=keras.initializers.TruncatedNormal(),kernel_regularizer=reg)(w)
            w=Dense(aq,activation=act,use_bias=False,kernel_initializer=keras.initializers.TruncatedNormal(),kernel_regularizer=reg)(w)
        w=Dense(q[-1],activation="linear",use_bias=False,kernel_initializer=keras.initializers.TruncatedNormal(),kernel_regularizer=reg)(w)
        m=Model(inn,w,name="oneoff")
        zero=K.ones_like(w)*mean
        loss=mse(w,zero)
        loss=K.mean(loss)
        m.add_loss(loss)
        m.compile(Adam(lr=hyper["lr"]))
        return m
    
    l=[dim for i in range(hyper["depth"])]
    m=getmodel(l,reg=None,act="relu",mean=1.0)
    
    cb=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
                       keras.callbacks.TerminateOnNaN()]
    cb.append(keras.callbacks.ModelCheckpoint(f"{pth}/model.tf", monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True))
    
    m.summary()
    
    #train the model    
    h=m.fit(train,None,
            epochs=500,
            batch_size=hyper["batch"],
            validation_split=0.25,
            verbose=1,
            callbacks=cb)

    #predict the output of our datasets 
    pain=m.predict(train)
    p=m.predict(t)
    w=m.predict(at)
   
    #average out the last dimension, to get one value for each samples 
    ppain=np.mean(pain,axis=-1)
    pp=np.mean(p,axis=-1)
    ww=np.mean(w,axis=-1)
    

    from sklearn.metrics import roc_auc_score as auc
   
    #calculate the mean prediction (q in the paper) 
    m=np.mean(ppain)

    #and the deviation of each to the mean
    pd=np.abs(pp-m)#if this worked, the values in the array pd should be much smaller
    wd=np.abs(ww-m)#than in the array wd
    y_score=np.concatenate((pd,wd))
    y_true=np.concatenate((np.zeros_like(pp),np.ones_like(ww)))
    
    #calculate auc score of a single model
    auc_score=auc(y_true,y_score)
    print(f"reached auc of {auc_score}")
    
    #and save the necessary results for merge.py to combine the submodel predictions into an ensemble
    np.savez_compressed(f"{pth}/result.npz",y_true=y_true,y_score=y_score,to_use=to_use)
    
    
    
    
    
if __name__ == "__main__":
    #train many models
    #this allows calling main.py with two arguments
    #python3 main.py [first model index] [second model index]
    i0=0
    i1=rounds
    if len(sys.argv)>1:
        i0=int(sys.argv[1])
        i1=i0+1
    if len(sys.argv)>2:
        i1=int(sys.argv[2])

    for i in range(i0,i1):
        train(i)
    
    
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
