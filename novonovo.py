import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tsfel
import csv
from sklearn import datasets, svm, ensemble, metrics
from sklearn.model_selection import train_test_split
from bitalino import BITalino

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
def signalParts(df, threshold):
    for i in range(len(df)):
        if df[i] > (1.1 * threshold):
            return df[i:(i + 500)], df[(i + 500):(i + 2000)]
#Cuts part of the signal
def signalParts2(df, threshold):
    return df[0: 500], df[500:2000]

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

clf.fit(xTestList, yTestList)

macAddress = "20:18:05:28:73:33"

#batteryThreshold = 30
acqChannels = [0]
samplingRate = 1000
nSamples = 500

# Connect to BITalino
device = BITalino(macAddress)

# Set battery threshold
#device.battery(batteryThreshold)

# Read BITalino version
print(device.version())

# Start Acquisition
device.start(samplingRate, acqChannels)

def is_relaxed(df, threshold):
    for i in abs(df):
        if i > (1.1 * threshold):
            return False
    else:
        return True

turnOff = False
window = np.zeros(5000)

window = np.zeros(2000)
relaxWindow = []
nSig=0
activated=False
while True:
    # Read samples
    sample = device.read(1)[0,5]
    if activated==False:
        #print(abs(sample-512), '\t',1.2*threshold)
        if abs(sample-512)>=1.1*threshold:
            window[nSig]=sample
            activated=True
            nSig+=1
    else:
        if nSig==2000:
            window-=window.mean()
            sigAc,sigPos=signalParts2(window,threshold)
            features=featureExtraction(sigAc,sigPos)
            predicted=clf.predict(features.reshape(1,-1))
            print('1 - ',predicted)
            nSig+=1
        elif nSig<2000:
            window[nSig]=sample
            nSig+=1
        else:
            #print(sample)
            relaxWindow.append(sample)
            window=np.zeros(2000)
            if len(relaxWindow)>500:
                print(max(np.abs(np.array(relaxWindow)-512)))
                if max(np.abs(np.array(relaxWindow)-512))<0.9*threshold:
                    activated=False
                    predicted='Relaxado'
                    print('2 - ', predicted)
                    nSig=0
                relaxWindow.pop(0)
        


# Stop acquisition
device.stop()

# Close connection
device.close()