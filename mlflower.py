# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 18:04:34 2021

@author: gauri
"""

# Required Packages
from sklearn import datasets		# To Get iris dataset
from sklearn import svm, metrics    			# To fit the svm classifier
from sklearn.model_selection import train_test_split

# import iris data to model Svm classifier
iris_dataset = datasets.load_iris()
#Iris dataset description Python
print("Iris data set Description :: ", iris_dataset['DESCR'])
print("Iris feature data :: ", iris_dataset['data'])
print("Iris target :: ", iris_dataset['target'])
X = iris_dataset.data  # we only take the Sepal two features.
y = iris_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)

# SVC with linear kernel
SVM = svm.SVC(kernel='linear')
classifier = SVM.fit(X_train, y_train)
predicted = classifier.predict(X_test)
expected = y_test
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
import pickle
filename = 'SVM.pkl'
pickle.dump(classifier, open(filename, 'wb'))

from sklearn import tree
#DT
DT = tree.DecisionTreeClassifier()
classifier = DT.fit(X_train, y_train)
predicted = classifier.predict(X_test)
expected = y_test
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
import pickle
filename = 'DT.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#KNN
from sklearn.neighbors import NearestCentroid
NB = NearestCentroid()
classifier = NB.fit(X_train, y_train)
predicted = classifier.predict(X_test)
expected = y_test
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
import pickle
filename = 'KNN.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#NB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
classifier = gnb.fit(X_train, y_train)
predicted = classifier.predict(X_test)
expected = y_test
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
import pickle
filename = 'NB.pkl'
pickle.dump(classifier, open(filename, 'wb'))