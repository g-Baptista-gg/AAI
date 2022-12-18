import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tsfel
#from bitalino import BITalino
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from sklearn import datasets, svm, metrics,ensemble
from sklearn.model_selection import train_test_split



cfg = tsfel.get_features_by_domain()




#Cuts part of the signal
def cut(df, threshold):
    for i in range(len(df)):
        if df[i] > threshold:
            return df[(i + 500):]

xTestList = []
yTestList = []
maxima = []



features_to_save=["0_Sum absolute diff","0_Median absolute diff","0_Signal distance","0_FFT mean coefficient_177","0_FFT mean coefficient_194","0_Wavelet variance_0","0_Wavelet standard deviation_0","0_Wavelet energy_0","0_FFT mean coefficient_58","0_LPCC_11"]

def getTestFeatures(nameClass, xTestList, yTestList, maxima):
    global threshold
    files = os.listdir(os.getcwd() + '/' + nameClass)[1::2]
    classData = [pd.read_csv(os.getcwd() + '/' + nameClass + '/' + i, header = 3, delimiter = '\t') for i in files]
    for i in classData:
        df = np.array(i.iloc[:, 5])
        df = df - df.mean()
        if nameClass == "Relaxado":
            maxima.append(max(df))
            X = tsfel.time_series_features_extractor(cfg, df,fs=1000)
        else:
            X = tsfel.time_series_features_extractor(cfg, cut(df, threshold),fs=1000)
        
        features_to_save_bool=[]
        for i in range(len(X.columns)):
            if X.columns[i] in features_to_save:
                features_to_save_bool.append(1)
            else:
                features_to_save_bool.append(0)
            
        X = X.values.tolist()
        X=np.array(X[0])
        #X=X[features_to_save_bool]
        yTestList.append(nameClass)
        xTestList.append(X)

Classes = ['Relaxado', 'Pedra', 'Papel', 'Tesoura']
for i in Classes:
    getTestFeatures(i, xTestList, yTestList, maxima)
    if i == "Relaxado":
        threshold = max(maxima)

cm = np.zeros((4, 4))
#clf = svm.SVC(kernel='rbf')
clf=ensemble.RandomForestClassifier()
#clf=ensemble.GradientBoostingClassifier()
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(xTestList, yTestList, test_size = 0.33, shuffle = True)
    #print(y_test)

    
    #clf.fit(xTestList, yTestList)
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    cm += disp.confusion_matrix
    #disp.figure_.suptitle("Confusion Matrix")
#print(disp.confusion_matrix())
for i in range(len(cm)):
    cm[i]/=cm[i].sum()
print(f"Confusion matrix:\n{cm}")


#