# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 19:08:14 2021

@author: gauri
"""

import pickle
import numpy as np
DT_model=pickle.load(open('DT.pkl', 'rb'))
KNN_model=pickle.load(open('KNN.pkl', 'rb'))
NB_model=pickle.load(open('NB.pkl', 'rb'))
SVM_model=pickle.load(open('SVM.pkl', 'rb'))
feature_list=[0,0,0,0,1,11,0,0,0,0,0,0,7,8,0,0,0,0,0,1,13,6,2,2,0,0,0,7,15,0,9,8,0,0,5,16,10,0,16,6,0,0,4,15,16,13,16,1,0,0,0,0,3,15,10,0,0,0,0,0,2,16,4,0]
single_pred = np.array(feature_list).reshape(1,-1)
clas=['0','1','2','3','4','5','6','7','8','9']
print(clas[int(DT_model.predict(single_pred))])
print(clas[int(KNN_model.predict(single_pred))])
print(clas[int(NB_model.predict(single_pred))])
print(clas[int(SVM_model.predict(single_pred))])