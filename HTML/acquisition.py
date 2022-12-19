import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tsfel
import csv
from sklearn import datasets, svm, ensemble, metrics
from sklearn.model_selection import train_test_split
from bitalino import BITalino
import time

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

clf.fit(xTestList, yTestList)

# The macAddress variable on Windows can be "XX:XX:XX:XX:XX:XX" or "COMX"
# while on Mac OS can be "/dev/tty.BITalino-XX-XX-DevB" for devices ending with the last 4 digits of the MAC address or "/dev/tty.BITalino-DevB" for the remaining
macAddress = "20:18:05:28:73:33"

# This example will collect data for 5 sec.
running_time = 5

batteryThreshold = 30
acqChannels = [0]
samplingRate = 1000
nSamples = 5000
digitalOutput_on = [1, 1]
digitalOutput_off = [0, 0]

# Connect to BITalino
device = BITalino(macAddress)

# Set battery threshold
device.battery(batteryThreshold)

# Read BITalino version
print(device.version())

# Start Acquisition
device.start(samplingRate, acqChannels)

sample = []

start = time.time()
end = time.time()
while (end - start) < running_time:
    # Read samples
    sample = device.read(nSamples)
    #print(device.read(nSamples))
    end = time.time()

# Turn BITalino led and buzzer on
#device.trigger(digitalOutput_on)

signal = sample[:, 5]
signal= np.array([float(i) for i in signal])

signal -= signal.mean()

act, pos = signalParts(signal, threshold)

ft = featureExtraction(act, pos)

predicted = clf.predict(ft.reshape(1,-1))
print(predicted)

#print(sample)

#plt.ylim(0, 1024)

# Script sleeps for n seconds
time.sleep(running_time)

# Turn BITalino led and buzzer off
#device.trigger(digitalOutput_off)

# Stop acquisition
device.stop()

# Close connection
device.close()
