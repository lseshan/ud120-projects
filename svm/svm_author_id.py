#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
sys.path.append("../choose_your_own")
import class_vis as cv


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
#clf = SVC(kernel='linear')
#clf = SVC(C=1.0,kernel='rbf') #.61
#clf = SVC(C=10.0, kernel='rbf') #.61
#1000 0.82
#10000 0.89
clf = SVC(C=10000.0, kernel='rbf') 


t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"
acc = accuracy_score(pred, labels_test)
print(acc)
print(pred[10], pred[26], pred[50])
print(len(pred))
count = 0
for i in range(len(pred)):
    if pred[i] is 1:
        count+=1
print(count)
#cv.prettyPicture(clf, features_test, labels_test)
#########################################################
### your code goes here ###

#########################################################


