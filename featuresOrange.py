import numpy as np
import pandas as pd
import os
import csv
from featureExtrationEMG import featureExtraction

feature_csv = open('featuresOrange.csv','w')
writer = csv.writer(feature_csv, lineterminator = '\n')

writer.writerow(['Feature', 'Absolute AUC Act', 'Absolute Mean Act', 'Mean Absolute Diff Act', 'Square AUC Act', 'Absolute AUC Post', 'Absolute Mean Post', 'Mean Absolute Diff Post', 'Square AUC Post', 'Variance Post', 'RMS Post', 'Median Freq Post'])

#Cuts part of the signal
def signalParts(df, threshold):
    for i in range(len(df)):
        if df[i] > (1.1 * threshold):
            return df[i:(i + 500)], df[(i + 500):(i + 2000)]
    else:
        return df[:500],df[500:2000]

maxima = []
threshold = 0

def getTestFeatures(nameClass, maxima):
    global threshold
    files = os.listdir(os.getcwd() + '/' + nameClass)[1::2]
    classData = [pd.read_csv(os.getcwd() + '/' + nameClass + '/' + i, header = 3, delimiter = '\t') for i in files]
    for i in classData:
        df = np.array(i.iloc[:, 5])
        df = df - df.mean()
        activation, postActivation = signalParts(df, threshold)
        if nameClass == "Relaxado":
            maxima.append(max(abs(df)))
        X = featureExtraction(activation, postActivation)
        Y = [nameClass]
        for i in X:
            Y.append(i)
        writer.writerow(Y)

Classes = ['Relaxado', 'Pedra', 'Papel', 'Tesoura']
for i in Classes:
    getTestFeatures(i, maxima)
    if i == "Relaxado":
        threshold = max(maxima)