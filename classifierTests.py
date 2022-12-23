import numpy as np
import pandas as pd
import os
from sklearn import ensemble, metrics
from sklearn.model_selection import train_test_split
from featureExtrationEMG import featureExtraction

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
        xTestList.append(featureExtraction(activation, postActivation))
        yTestList.append(nameClass)
    return xTestList,yTestList

Classes = ['Relaxado', 'Pedra', 'Papel', 'Tesoura']
for i in Classes:
    xTestList,yTestList = getTestFeatures(i, xTestList, yTestList, maxima)

    if i == "Relaxado":
        threshold = max(maxima)

clf = ensemble.RandomForestClassifier()

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
    cm[i] /= cm[i].sum()
print(cm)