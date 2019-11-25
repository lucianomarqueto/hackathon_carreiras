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
from sklearn.utils import shuffle

path = "E:/_python_projects/hackathon_carreiras/data/"
path_out = "E:/_python_projects/hackathon_carreiras/data/"
#path = "D:/python/hackathon_carreiras/data/"
#path_out = "D:/python/hackathon_carreiras/data/"

#Carrega Dataset
df_file = pd.read_csv(path + "bank-additional-full.csv",";")
df_file = shuffle(df_file)

#Remove atributo não disponivel para previsão conforme documentação do dataset
df_file = df_file.drop(["duration"],axis=1)

#analise estatistica
describe = df_file.describe()
corr = df_file.corr()

#remover colunas com alta correlação
#Apos remoção desses elementos foi possível obter manter o mesmo resultado
df_file = df_file.drop(["pdays"],axis=1)
df_file = df_file.drop(["euribor3m"],axis=1)
df_file = df_file.drop(["nr.employed"],axis=1)
df_file = df_file.drop(["emp.var.rate"],axis=1)


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
df_numerico['previous'] = df_numerico['previous'].clip(0,1)
df_numerico.hist(figsize=(10,10));
plt.show()


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


# Split the dataset in training and test set:
df_train, df_test, y_train, y_test = train_test_split(
    df_final, df_label, test_size=0.4)


#Balanceamento

#Foram testado diversos tipos de balanceamento para ver qual aprensentaria o melhor resultado


#from imblearn.under_sampling import ClusterCentroids
#cc = ClusterCentroids(ratio={0: 10})
#X, y = cc.fit_sample(df_train, y_train['y'])

#from imblearn.under_sampling import TomekLinks
#tl = TomekLinks(return_indices=True, ratio='majority')
#X, y, id_tl = tl.fit_sample(df_train, y_train['y'])


#from imblearn.under_sampling import RandomUnderSampler
#rus = RandomUnderSampler(return_indices=True)
#X, y, id_rus = rus.fit_sample(df_train, y_train['y'])

#from imblearn.over_sampling import RandomOverSampler
#ros = RandomOverSampler()
#X, y = ros.fit_sample(df_train, y_train['y'])


from imblearn.combine import SMOTETomek
smt = SMOTETomek(ratio='auto')
X, y = smt.fit_sample(df_train, y_train['y'])

cols = df_train.columns
df_train = pd.DataFrame(X)
df_train.columns = cols
cols = y_train.columns
y_train = pd.DataFrame(y)
y_train.columns = cols


# =============================================================================

######################

#df_final = pd.merge(df_final,df_label,left_index=True, right_index=True)  


#####################
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


names = [#"Nearest Neighbors", 
         "Linear SVM", 
         #"RBF SVM", 
         #"Gaussian Process",
         "Decision Tree", 
         "Random Forest", 
         "Neural Net", 
         "AdaBoost",
         #"Naive Bayes", 
         #"QDA"
         ]

classifiers = [
    #KNeighborsClassifier(3),
    SVC(kernel="linear", C=1.0),
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=7),
    RandomForestClassifier(max_depth=7, n_estimators=100),
    MLPClassifier(alpha=0.3, max_iter=2000),
    AdaBoostClassifier(),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis()
    ]

for name, clf in zip(names, classifiers):
    print("###############################")
    print(name)
    clf = clf.fit(df_train, y_train)

    predicted = clf.predict(df_test)
    np.mean(predicted == y_test['y'])  

    # Print the classification report
    print(metrics.classification_report(y_test, predicted,
                                    target_names=['no','yes']))

    # Plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    print(cm)