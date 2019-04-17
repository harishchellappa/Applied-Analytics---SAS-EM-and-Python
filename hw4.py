# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:28:56 2019

@author: haris
"""
from AdvancedAnalytics import DecisionTree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_graphviz 
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

#10-fold CV 
score_list = ['accuracy', 'recall', 'precision', 'f1']
search_depths = [5,6,7,8,10,12,15,20,25]
for d in search_depths:
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=d, min_samples_split=5, min_samples_leaf=5)
    mean_score = []
    std_score = []
    print("max_depth=", d)
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        dtc_10 = cross_val_score(dtc, X, y, scoring=s, cv=10)
        mean = dtc_10.mean()
        std = dtc_10.std()
        mean_score.append(mean)
        std_score.append(std)
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))

#the optimum decision tree
dtc = DecisionTreeClassifier(criterion='gini', max_depth=5,min_samples_split=5, min_samples_leaf=5)
X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)
dtc = dtc.fit(X_train,y_train)

classes = [ 'good','bad']
col = rie.col
col.remove('good_bad') 
DecisionTree.display_importance(dtc, col)
DecisionTree.display_binary_split_metrics(dtc, X_train, y_train, X_validate, y_validate)

'''
### TEXTBOOK WAY OF GETTINGA TREE - NOT WORKING

dot_data = export_graphviz(dtc, filled=True, rounded=True, \
                           class_names=classes, feature_names = col, out_file=None)
#write tree to png file 'homework_tree' 
graph_png = graph_from_dot_data(dot_data) 
graph_path = r'C:\Users\haris\Documents\stat 656\week4' 
graph_png.write_png(r"C:\Users\haris\Documents\stat 656\week4\homework_tree.png") 
graph_pdf = graphviz.Source(dot_data) 
graph_pdf.view("tree") 
'''
# from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
featureNames=df_rie[0:68]
export_graphviz(dtc,out_file=dot_data, class_names= ['1:Good','0:Bad'], filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

