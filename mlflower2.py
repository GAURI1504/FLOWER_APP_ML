# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 18:27:39 2021

@author: gauri
"""

import pickle
import numpy as np

SVM_model=pickle.load(open('SVM.pkl', 'rb'))
sl=input('Sepal Length=')
sw=input('Sepal Width=')
pl=input('Petal Length=')
pw=input('Petal Width=')
feature_list=[sl,sw,pl,pw]
single_pred = np.array(feature_list).reshape(1,-1)
clas=['setosa','versicolor','virginica']

print(clas[int(SVM_model.predict(single_pred))])