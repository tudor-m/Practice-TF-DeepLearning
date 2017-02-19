
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os

tf.logging.set_verbosity(verbosity=tf.logging.INFO)

train_file = "SimpleProd.csv"
test_file = "SimpleProd.test.csv"
COLUMNS = ["x1","x2","y"]
FEATURES = ["x1","x2"]
LABEL = "y"

df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
