import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tsfel
import csv
from sklearn import datasets, svm, ensemble, metrics
from sklearn.model_selection import train_test_split

def featureExtraction(activation, postActivation):
    features = np.array([tsfel.feature_extraction.features.auc(abs(activation), 1000),
    tsfel.feature_extraction.features.calc_mean(abs(activation)),
    tsfel.feature_extraction.features.mean_abs_diff(activation),
    tsfel.feature_extraction.features.auc((activation ** 2), 1000),
    #tsfel.feature_extraction.features.calc_var(activation),
    #tsfel.feature_extraction.features.rms(activation),
    #tsfel.feature_extraction.features.median_frequency(activation, 1000),
    tsfel.feature_extraction.features.auc(abs(postActivation), 1000),
    tsfel.feature_extraction.features.calc_mean(abs(postActivation)),
    tsfel.feature_extraction.features.mean_abs_diff(abs(postActivation)),
    tsfel.feature_extraction.features.auc((postActivation ** 2), 1000),
    tsfel.feature_extraction.features.calc_var(postActivation),
    tsfel.feature_extraction.features.rms(postActivation),
    tsfel.feature_extraction.features.median_frequency(postActivation, 1000)])
    #waveform length!!!
    #autoregressive coefficients!!!
    return features

feature_csv = open('featuresPlanoB.csv','w')
writer = csv.writer(feature_csv, lineterminator = '\n')

writer.writerow(['Feature', 'Absolute AUC Act', 'Absolute Mean Act', 'Mean Absolute Diff Act', 'Square AUC Act', 'Absolute AUC Post', 'Absolute Mean Post', 'Mean Absolute Diff Post', 'Square AUC Post', 'Variance Post', 'RMS Post', 'Median Freq Post'])
#'Variance Act', 'RMS Act', 'Median Freq Act',

#Cuts part of the signal
def signalParts(df, threshold):
    for i in range(len(df)):
        if df[i] > (1.1 * threshold):
            return df[i:(i + 500)], df[(i + 500):(i + 2000)]

xTestList = []
yTestList = []
maxima = []
threshold = 0

def getTestFeatures(nameClass, xTestList, yTestList, maxima):
    global threshold
    files = os.listdir(os.getcwd() + '/' + nameClass)[1::2]
    classData = [pd.read_csv(os.getcwd() + '/' + nameClass + '/' + i, header = 3, delimiter = '\t') for i in files]
    for i in classData:
        df = np.array(i.iloc[:, 5])
        df = df - df.mean()
        activation, postActivation = signalParts(df, threshold)
        if nameClass == "Relaxado":
            maxima.append(max(abs(df)))
        #xTestList.append(featureExtraction(activation, postActivation))
        #yTestList.append(nameClass)
        xTestList.append(featureExtraction(activation, postActivation))
        yTestList.append(nameClass)
        #for i in X:
        #    Y.append(i)
        #writer.writerow(Y)
    return xTestList,yTestList

Classes = ['Relaxado', 'Pedra', 'Papel', 'Tesoura']
for i in Classes:
    xTestList,yTestList=getTestFeatures(i, xTestList, yTestList, maxima)

    if i == "Relaxado":
        threshold = max(maxima)

clf = ensemble.RandomForestClassifier()
#clf=svm.SVC()

cm = np.zeros((4,4))
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(xTestList, yTestList, test_size = 0.33, shuffle = True)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    cm += disp.confusion_matrix
    #disp.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")
for i in range(len(cm)):
    cm[i]/=cm[i].sum()
print(cm)