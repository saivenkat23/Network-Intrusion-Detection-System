import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (f1_score, accuracy_score)
import random as rd

traindata = pd.read_csv('E:\Intrusion-Detection-Systems-master\Intrusion-Detection-Systems-master\kddtrain.csv', header=None)
testdata = pd.read_csv('E:\Intrusion-Detection-Systems-master\Intrusion-Detection-Systems-master\kddtest.csv', header=None)

X = traindata.iloc[:, 1:42]
X
Y = traindata.iloc[:, 0]
Y
C = testdata.iloc[:, 0]
C
T = testdata.iloc[:, 1:42]
T

#normalisation
scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)


traindata = np.array(trainX)
trainlabel = np.array(Y)

testdata = np.array(testT)
testlabel = np.array(C)


#LOGISTICAL REFGRESSION
model = LogisticRegression()
model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
f1 = f1_score(y_train1, y_pred)


print("accuracy")
print("%.3f" % accuracy)
print("f1score")
print("%.3f" % f1)

for i in range(311029):
    predict_rand = rd.randint(0, 311029)
    print("Checking for %d index in predicted array" % predict_rand)
    if predicted[predict_rand] == 0:
        print("The network is SAFE to use.")
        break
    else:
        print("The Network is not at all SAFE.")
        break
