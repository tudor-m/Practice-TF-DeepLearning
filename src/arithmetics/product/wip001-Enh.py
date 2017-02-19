'''
Created on Jan 28, 2017

@author: user
'''

if __name__ == '__main__':
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import random
    import os
    
    tf.logging.set_verbosity(verbosity=tf.logging.INFO)
    
    train_file = "SimpleProdEnhanced.csv"
    test_file = "SimpleProd.Enhanced.test.csv"
    COLUMNS = ["x1","x2","y"]+["x1_2","x2_2","x12","x12_2"]
    FEATURES = ["x1","x2"]+["x1_2","x2_2","x12","x12_2"]
    LABEL = "y"
    
    df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
    df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
    
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    
    regressor = tf.contrib.learn.DNNRegressor(feature_columns = feature_cols,
                                              hidden_units = [22,256,24,23,21,21,2,2],
                                              model_dir = "./tf-model")
    random.seed(100)

    
    def input_fn(data_set):
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        return feature_cols, labels
    
    regressor.fit(input_fn=lambda:input_fn(df_train),steps = 2000)
    
    eval = regressor.evaluate(input_fn=lambda: input_fn(df_test),steps = 1)
    
    y    = regressor.predict(input_fn=lambda: input_fn(df_test))
    y_val = np.empty(0)
    for k in range(0,1+y.__sizeof__()):
        y_val = np.append(y_val,y.next())
    
    print "Hello there!"