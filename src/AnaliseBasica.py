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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

path = "E:/_python_projects/hackathon_carreiras/data/"
path_out = "E:/_python_projects/hackathon_carreiras/data/"
#path = "D:/python/hackathon_carreiras/data/"
#path_out = "D:/python/hackathon_carreiras/data/"

#Carrega Dataset
df_file = pd.read_csv(path + "bank-additional-full.csv",";")

#Remove atributo não disponivel para previsão conforme documentação do dataset
df_file = df_file.drop(["duration"],axis=1)

#analise estatistica
describe = df_file.describe()
corr = df_file.corr()

#Analisar Balanceamento
target_count = df_file.y.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)');
plt.show()

#split df nominal e numerico e label
df_obj = df_file.select_dtypes(include=[object])
columns_obj = df_obj.columns
df_nominal = df_obj.drop(["y"],axis=1)
df_numerico = df_file.drop(columns_obj,axis=1)
columns_obj = columns_obj.drop('y')
df_label = df_obj.drop(columns_obj,axis=1)

#Converte yes - 1 no - 0
df_label["y"] = df_label.apply(lambda x : 1 if x['y'] == 'yes' else 0, axis = 1)


#Removendo outliers
df_numerico['age'] = df_numerico['age'].clip(17,70)
df_numerico['campaign'] = df_numerico['campaign'].clip(0,10)
df_numerico['previous'] = df_numerico['previous'].clip(0,2)
df_numerico.hist(figsize=(10,10));

#for col in df_nominal.columns:
#    df_nominal[col].value_counts().plot(kind='bar', title='Count ('+col+')');
#    plt.show()

#one hot encode atributos nominais
df_nominal = pd.get_dummies(df_nominal)

#scaling
for col in df_numerico.columns:             
    max_c = df_numerico[col].max()
    min_c = df_numerico[col].min()
    desloc = (min_c*-1) if min_c < 0 else min_c
    max_c = max_c + desloc
    df_numerico[col] = df_numerico.apply(lambda x: (x[col]+desloc)/max_c  , axis = 1)        


#Merge
df_final = pd.merge(df_nominal,df_numerico,left_index=True, right_index=True)  

#Balanceamento
def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

pca = PCA(n_components=2)
df_final_x = pca.fit_transform(df_final)
plot_2d_space(df_final_x, df_label['y'], 'Imbalanced dataset (2 PCA components)')


from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(ratio={0: 10})
X_cc, y_cc = cc.fit_sample(df_final, df_label['y'])
plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')
X_resampled = pd.DataFrame(X_cc)
X_resampled.columns = df_final.columns


from imblearn.under_sampling import TomekLinks
tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(df_final, df_label['y'])
print('Removed indexes:', id_tl)
plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(return_indices=True)
X_tl, y_tl, id_rus = rus.fit_sample(df_final, df_label['y'])


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_tl, y_tl = ros.fit_sample(df_final, df_label['y'])

print(X_tl.shape[0] - df_final.shape[0], 'new random picked points')

plot_2d_space(X_tl, y_tl, 'Random over-sampling')

X_resampled = pd.DataFrame(X_tl)
X_resampled.columns = df_final.columns
y_resampled = pd.DataFrame(y_tl)
y_resampled.columns = df_label.columns
target_count = y_resampled.y.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)');
plt.show()

######################

#df_final = pd.merge(df_final,df_label,left_index=True, right_index=True)  

# Split the dataset in training and test set:
df_train, df_test, y_train, y_test = train_test_split(
    df_final, df_label, test_size=0.4)

#treinar modelo
clf = MultinomialNB().fit(df_train, y_train)

#Testar
predicted = clf.predict(df_test)
np.mean(predicted == y_test['y'])  

# Print the classification report
print(metrics.classification_report(y_test, predicted,
                                    target_names=['no','yes']))

# Plot the confusion matrix
cm = metrics.confusion_matrix(y_test, predicted)
print(cm)


