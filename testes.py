import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tsfel
from bitalino import BITalino
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

cfg = tsfel.get_features_by_domain()

#Cuts part of the signal
def cut(df, threshold):
    for i in range(len(df)):
        if df[i] > threshold:
            return df[(i + 500):]

xTestList = []
yTestList = []
maximus = []

def getTestFeatures(nameClass, xTestList, yTestList, maximus):
    global threshold
    files = os.listdir(os.getcwd() + '/' + nameClass)[1::2]
    classData = [pd.read_csv(os.getcwd() + '/' + nameClass + '/' + i, header = 3, delimiter = '\t') for i in files]
    for i in classData:
        df = np.array(i.iloc[:, 5])
        df = df - df.mean()
        if nameClass == "Relaxado":
            maximus.append(max(abs(df)))
        X = tsfel.time_series_features_extractor(cfg, df)
        #else:
        #    X = tsfel.time_series_features_extractor(cfg, cut(df, threshold))
        X = X.values.tolist()
        yTestList.append(nameClass)
        xTestList.append(X[0])

Classes = ['Relaxado', 'Pedra', 'Papel', 'Tesoura']
for i in Classes:
    getTestFeatures(i, xTestList, yTestList, maximus)
    if i == "Relaxado":
        threshold = max(maximus)

clf = SVC(gamma = 'auto')
clf_NB = GaussianNB()
clf_kNN = KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(xTestList, yTestList, test_size = 0.33)

cm1 = np.zeros((4, 4))
cm2 = np.zeros((4, 4))
cm3 = np.zeros((4, 4))

for i in range(10):

    #clf.fit(xTestList, yTestList)
    clf.fit(X_train, y_train)
    clf_NB.fit(X_train, y_train)
    clf_kNN.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    predicted_NB = clf_NB.predict(X_test)
    predicted_kNN = clf_kNN.predict(X_test)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    #disp.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")

    disp_NB = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_NB)
    #disp_NB.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp_NB.confusion_matrix}")

    disp_kNN = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_kNN)
    #disp_kNN.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp_kNN.confusion_matrix}")

    cm1 += disp.confusion_matrix
    cm2 += disp_NB.confusion_matrix
    cm3 += disp_kNN.confusion_matrix

print(f"Confusion matrix:\n{cm1}")
print(f"Confusion matrix:\n{cm2}")
print(f"Confusion matrix:\n{cm3}")