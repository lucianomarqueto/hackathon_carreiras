# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:05:29 2019

@author: luciano



O Programa pode ser executado de uma unica vez é apenas necessario configuar a variavel path

O programa vai tratar os dados
Criar o modelo 
Salvar o modelo
inferir o valor de novos dados

Todo codigo utilizado para analise esta incluso e ou comentado
na pasta resutado encotrase o resultado de diversos treinamentos que foram 
efetudos para identificar o algoritimo e recursos

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import imblearn

#path = "E:/_python_projects/hackathon_carreiras/data/"
#path_out = "E:/_python_projects/hackathon_carreiras/model/"
path = "D:/python/hackathon_carreiras/data/"
path_out = "D:/python/hackathon_carreiras/model/"

#valores obtidos depois da analise utlizado como base para o scaling
dic_scal = {'age':70,'campaign':10,'previous':1,'cons.price.idx':95,'cons.conf.idx':51}
colunas = ['job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'marital_unknown', 'education_basic.4y', 'education_basic.6y',
       'education_basic.9y', 'education_high.school', 'education_illiterate',
       'education_professional.course', 'education_university.degree',
       'education_unknown', 'default_no', 'default_unknown', 'default_yes',
       'housing_no', 'housing_unknown', 'housing_yes', 'loan_no',
       'loan_unknown', 'loan_yes', 'contact_cellular', 'contact_telephone',
       'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun',
       'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
       'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu',
       'day_of_week_tue', 'day_of_week_wed', 'poutcome_failure',
       'poutcome_nonexistent', 'poutcome_success', 'age', 'campaign',
       'previous', 'cons.price.idx', 'cons.conf.idx']

#Carrega Dataset
df_file = pd.read_csv(path + "bank-additional-full.csv",";")



#analise estatistica
describe = df_file.describe()
corr = df_file.corr()



def pre(df_file):
    
    #Remove atributo não disponivel para previsão conforme documentação do dataset
    df_file = df_file.drop(["duration"],axis=1)
    
    #remover colunas com alta correlação
    #Apos remoção desses elementos foi possível obter manter o mesmo resultado
    df_file = df_file.drop(["pdays"],axis=1)
    df_file = df_file.drop(["euribor3m"],axis=1)
    df_file = df_file.drop(["nr.employed"],axis=1)
    df_file = df_file.drop(["emp.var.rate"],axis=1)
    
    
    #Analisar Balanceamento
    #target_count = df_file.y.value_counts()
    #print('Class 0:', target_count[0])
    #print('Class 1:', target_count[1])
    #print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
    #target_count.plot(kind='bar', title='Count (target)');
    #plt.show()
    
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
    def clip(df):
        df['age'] = df['age'].clip(17,70)
        df['campaign'] = df['campaign'].clip(0,10)
        df['previous'] = df['previous'].clip(0,1)
        #df_numerico.hist(figsize=(10,10));
        #plt.show()
        return df
    df_numerico = clip(df_numerico)
    
    #for col in df_nominal.columns:
    #    df_nominal[col].value_counts().plot(kind='bar', title='Count ('+col+')');
    #    plt.show()
    
    #one hot encode atributos nominais
    df_nominal = pd.get_dummies(df_nominal)
    
    df_numerico['cons.conf.idx']  = df_numerico.apply(lambda x: x['cons.conf.idx']*-1  , axis = 1) 
    #scaling
    for col in df_numerico.columns:             
        df_numerico[col] = df_numerico.apply(lambda x: x[col]/dic_scal[col]  , axis = 1)        
    
    
    #Merge
    return pd.merge(df_nominal,df_numerico,left_index=True, right_index=True) , df_label

df_final, df_label = pre(df_file)

# Split the dataset in training and validate test set:
df_train, df_test, y_train, y_test = train_test_split(
    df_final, df_label, test_size=0.4)

df_test, df_val, y_test, y_val = train_test_split(
    df_test, y_test, test_size=0.5)



#####################################
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

#Foram testados diversos algoritimos de classificacao
#O que teve melhor desenpenho considerando a metrica F1 SCORE foi a DecisionTreeClassifier(max_depth=8)

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

#
#names = [#"Nearest Neighbors", 
#         "Linear SVM", 
#         #"RBF SVM", 
#         #"Gaussian Process",
#         "Decision Tree", 
#         "Random Forest", 
#         "Neural Net", 
#         "AdaBoost",
#         #"Naive Bayes", 
#         #"QDA"
#         ]
#
#classifiers = [
#    #KNeighborsClassifier(3),
#    SVC(kernel="linear", C=1.5),
#    #SVC(gamma=2, C=1),
#    #GaussianProcessClassifier(1.0 * RBF(1.0)),
#    DecisionTreeClassifier(max_depth=8),
#    RandomForestClassifier(max_depth=7, n_estimators=200),
#    MLPClassifier(alpha=0.1, max_iter=2000),
#    AdaBoostClassifier(),
#    #GaussianNB(),
#    #QuadraticDiscriminantAnalysis()
#    ]
#
#for name, clf in zip(names, classifiers):
#    print("###############################")
#    print(name)
#    clf = clf.fit(df_train, y_train)
#
#    predicted = clf.predict(df_test)
#    
#    # Print the classification report
#    print(metrics.classification_report(y_test, predicted,
#                                    target_names=['no','yes']))
#
#    # Plot the confusion matrix
#    cm = metrics.confusion_matrix(y_test, predicted)
#    print(cm)




#O resultado se mostrou consistente mesmo utilizando
#os dados de validacao 

clf = DecisionTreeClassifier(max_depth=8)
clf = clf.fit(df_train, y_train)
predicted = clf.predict(df_val)
# Print the classification report
print(metrics.classification_report(y_val, predicted,
                                    target_names=['no','yes']))

# Plot the confusion matrix
cm = metrics.confusion_matrix(y_val, predicted)
print(cm)
   

 
import pickle
# Salvar modelo
filename = path_out+'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))


# Carregar modelo e classificar alguns registros
loaded_model = pickle.load(open(filename, 'rb'))
df_file = pd.read_csv(path + "simple_exemple.csv",";")

#apenas para validacao do programa
#df_y = df_file.copy()
#columns_obj = df_file.columns.drop('y')
#df_y = df_y.drop(columns_obj,axis=1)
#df_y["y"] = df_y.apply(lambda x : 1 if x['y'] == 'yes' else 0, axis = 1)

    
df_file['y'] = 'unknow'
df_final, y = pre(df_file)
for col in colunas:
    if col not in df_final:
        df_final[col] = 0
        
predicted = clf.predict(df_final)

#print(metrics.classification_report(df_y, predicted))

print(predicted)
