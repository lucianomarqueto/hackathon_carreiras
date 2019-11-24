# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:05:29 2019

@author: luciano
"""

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

path = "D:/python/hackathon_carreiras/data/"
path_out = "D:/python/hackathon_carreiras/data/"


df_file = pd.read_csv(path + "bank-additional-full.csv",";")

df_file.apply()

describe = df_file.describe()

corr = df_file.corr()


df_obj = df_file.select_dtypes(include=[object])
df_obj.head(3)

# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()




# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
df_obj = df_file.apply(le.fit_transform)
df_obj.head()

le = preprocessing.LabelEncoder()

df_obj_2 = df_file.apply(le.fit_transform)
df_obj_2.head()


enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(df_obj_2)

# 3. Transform
onehotlabels = enc.transform(df_obj_2).toarray()
onehotlabels


def hotencode(df, col):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df[col])
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    
hotencode(df_file, 'marital')