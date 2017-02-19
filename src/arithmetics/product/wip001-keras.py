'''
Created on Feb 6, 2017

@author: user
'''
from keras.models import Sequential
from keras.layers import Dense,Activation
import pandas as pd
import numpy as np
import random
import os
from keras.optimizers import SGD

global df_train, df_test, COLUMNS, FEATURES, LABEL

def load_data():
    global df_train, df_test, COLUMNS, FEATURES, LABELS, df_train_norm, df_test_norm, norm_col
    PATH = "/Users/user/Projects/Practice/TF-DeepLearning/src/arithmetics/product/"
    train_file = "SimpleProdEnhanced.csv"
    test_file = "SimpleProdEnhanced.test.csv"
    IN_FEATURES = ["x1","x2"]
    ADDED_FEATURES = ["x1_2","x2_2","x12","x12_2"]
    TARGET_FEATURES = ["y"]
    COLUMNS = IN_FEATURES + TARGET_FEATURES + ADDED_FEATURES
    
    FEATURES = IN_FEATURES + ADDED_FEATURES
    LABELS = TARGET_FEATURES

    df_train = pd.read_csv(PATH+train_file, skipinitialspace=True)
    df_test = pd.read_csv(PATH+test_file, skipinitialspace=True)
    # Normalize the inputs:
    norm_train_col = df_train.mean()
    norm_test_col = df_train.mean()
    df_train_norm = df_train/norm_train_col
    df_test_norm = df_test/norm_test_col
    return

def keras_body():
    model = Sequential() # linear stack of layers
    activ = "tanh"
    activ = "relu"
    activ = "softmax"
    activ = "tanh"
    activ_list = ["softmax","softplus","softsign","relu","tanh","sigmoid","hard_sigmoid"]
    #activ_list = ["tanh"]
    grid_error = dict()
    grid_model = dict()
    grid_pred = dict()
    for momnt in [7, 7.1, 7.2, 7.3]:
        seed = 10
        np.random.seed(seed)
        for activ0 in ["relu"]:
            del(model)
            model = Sequential()
            # Input layer:
            model.add(Dense(input_dim=len(FEATURES) ,output_dim=32,bias=True))
            #model.add(Dense(input_dim=2 ,output_dim=32,bias=True))
            model.add(Activation(activ0))
            for activ1 in ["relu"]:
                # Hidden layers
                #hidden_size = [8]
                hidden_size = [32]
                for el in hidden_size:
                    model.add(Dense(output_dim=el,bias=True))
                    model.add(Activation("relu"))
                    #model.add(Dense(output_dim=el,bias=True))
                    #model.add(Activation(activ1))
                # Output layer
                out_dim = LABELS.__len__()
                model.add(Dense(output_dim=out_dim,bias=False))
                #model.add(Dense(output_dim=1,bias=False))
                model.add(Activation("relu"))
                mmt = int(momnt)/10. 
                sgd = SGD(lr=0.001, decay=1e-7, momentum = mmt, nesterov=True)
                #sgd = SGD(lr=0.001, decay=1e-3, momentum = mmt., nesterov=True)
                model.compile(loss='mae',optimizer=sgd)
                x_train = df_train.loc[:,FEATURES]
                y_train = df_train.loc[:,LABELS]#[:,LABELS]
                #x_train = df_train_norm.loc[:,FEATURES]
                #y_train = df_train_norm.loc[:,LABELS]#[:,LABELS]
                mfit = model.fit(np.asarray(x_train),np.asarray(y_train),nb_epoch=5000,batch_size=4,verbose=0)
                
                x_eval = df_test.loc[:,FEATURES]
                y_eval = df_test.loc[:,LABELS]
                #x_eval = df_test_norm.loc[:,FEATURES]
                #y_eval = df_test_norm.loc[:,LABELS]
                loss_and_metrics = model.evaluate(np.asarray(x_eval),np.asarray(y_eval),batch_size=1)
                #print(loss_and_metrics)
                
                pred = model.predict(x=np.asarray(x_eval), batch_size=1, verbose=0)
                k = activ0+"-"+activ1+"-"+str(momnt)
                grid_error[k] = loss_and_metrics
                grid_model[k] = mfit
                grid_pred[k] = pred
    for k in grid_model.keys():
        print(k)
        pred = grid_pred[k]
        print("max abs:")
        print(abs(pred-y_eval.values).max())

    #print(grid_pred)
    np.c_[y_eval,grid_pred['relu-relu-6']]
    print ("Hello! End of Task!")
    return



if __name__ == '__main__':
    load_data()
    keras_body()