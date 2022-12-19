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
from sklearn.ensemble import RandomForestClassifier

cfg = tsfel.get_features_by_tag(tag = 'emg')

def featureExtraction(signal):
    features = []

    fftMean = tsfel.feature_extraction.features.fft_mean_coeff(signal, 1000)
    features.append(fftMean[52])
    features.append(fftMean[69])
    features.append(fftMean[117])
    features.append(fftMean[177])
    features.append(fftMean[199])
    features.append(fftMean[218])
    features.append(tsfel.feature_extraction.features.spectral_roll_on(signal, 1000))

    histogram = tsfel.feature_extraction.features.hist(signal)
    features.append(histogram[8])
    features.append(tsfel.feature_extraction.features.distance(signal))
    features.append(tsfel.feature_extraction.features.sum_abs_diff(signal))

    return features

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
            xTestList.append(featureExtraction(df))
        #X = tsfel.time_series_features_extractor(cfg, df)
        else:
            xTestList.append(featureExtraction(cut(df, threshold)))
        #    X = tsfel.time_series_features_extractor(cfg, cut(df, threshold))
        #X = X.values.tolist()
        yTestList.append(nameClass)
        #xTestList.append(X[0])

Classes = ['Relaxado', 'Pedra', 'Papel', 'Tesoura']
for i in Classes:
    getTestFeatures(i, xTestList, yTestList, maximus)
    if i == "Relaxado":
        threshold = max(maximus)

clf = SVC(gamma = 0.1)
clf_rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=5)

X_train, X_test, y_train, y_test = train_test_split(xTestList, yTestList, test_size = 0.33)

cm1 = np.zeros((4, 4))

for i in range(10):

    #clf.fit(xTestList, yTestList)
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")

    cm1 += disp.confusion_matrix

    #disp.figure_.suptitle("Confusion Matrix")

print(f"Confusion matrix:\n{cm1}")

#plt.show()