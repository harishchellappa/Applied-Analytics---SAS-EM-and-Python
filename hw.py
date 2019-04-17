# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:42:14 2019

@author: haris
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:28:56 2019

@author: haris
"""
from AdvancedAnalytics import NeuralNetwork
from sklearn.tree import export_graphviz 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import cross_val_score 
from pydotplus import graph_from_dot_data
from sklearn.model_selection import train_test_split
import graphviz as show_tree 
import pandas as pd 
import numpy as np

df = pd.read_excel(r"C:\Users\haris\Documents\stat 656\week4\CreditHistory_Clean.xlsx")

attribute_map = {
        'age':['I', (19, 120)],
        'amount':['I', (0, 20000)],
        'checking':['N',(1,2,3,4)],
        'coapp':['N',(1, 2, 3)],
        'depends':['B',(1, 2)],
        'duration':['I',(1,72)],
        'employed':['N',(1,2,3,4,5)],
        'existcr':['N',(1,2,3,4)],
        'foreign':['B', (1,2)],
        'good_bad':['B', ('bad','good')],
        'history':['N', (0,1,2,3,4)],
        'housing':['N', (1,2,3)],
        'installp':['N', (1,2,3,4)],
        'job':['N', (1,2,3,4)],
        'marital':['N', (1,2,3,4)],
        'other':['N', (1,2,3)],
        'property':['N', (1,2,3,4)],
        'purpose':['N', ('0','1','2','3','4','5','6','8','9','X')], 
        'resident':['N', (1,2,3,4)],
        'savings':['N', (1,2,3,4,5)],
        'telephon':['B', (1,2)] }

rie = ReplaceImputeEncode(data_map=attribute_map, interval_scale=None, nominal_encoding='one-hot', display=True, drop = False) 
df_rie = rie.fit_transform(df)
print("\nData after replacing outliers, imputing missing and encoding:") 
print(df_rie.head())

#good_bad is the name of the binary target 
varlist = ['good_bad'] 
X = np.asarray(df_rie.drop(varlist, axis=1)) 
y = np.asarray(df_rie['good_bad']) 
np_y=np.ravel(y)
#4-fold CV 
score_list = ['accuracy', 'recall', 'precision', 'f1']
network_list = [(3),(11),(5,4),(6,5),(7,6),(8,7)]
# Scoring for Interval Prediction Neural Networks 
for nn in network_list:
    fnn = MLPClassifier(hidden_layer_sizes=nn, solver="lbfgs", activation="relu", \
                        max_iter=1000, random_state=12345)
    mean_score = [ ] 
    std_score = [ ] 
    print("\nDecision Tree with Hidden_Layer_Sizes=", nn) 
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean","Std. Dev.")) 
    for s in score_list:
        scores = cross_val_score(fnn,X,np_y,scoring=s,cv=4) 
        mean = scores.mean()
        std = scores.std()
        mean_score.append(scores.mean()) 
        std_score.append(scores.std()) 
        print("{:.<10s}{:>6.4f}{:>13.4f}".format(s,mean,std))

X_train, X_validate, y_train, y_validate = train_test_split(X,np_y,test_size = 0.3, \
                                                            random_state=12345)
fnn = MLPClassifier(hidden_layer_sizes=(6,5), solver="lbfgs", activation="relu", \
                        max_iter=1000, random_state=12345)  
fnn = fnn.fit(X_train, y_train)
NeuralNetwork.display_binary_split_metrics(fnn,X_train, y_train, X_validate, y_validate)    
