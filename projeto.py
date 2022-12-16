import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tsfel
from bitalino import BITalino
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

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
        df = abs(df - df.mean())
        if nameClass == "Relaxado":
            maximus.append(max(df))
            X = tsfel.time_series_features_extractor(cfg, df)
        else:
            X = tsfel.time_series_features_extractor(cfg, cut(df, threshold))
        X = X.values.tolist()
        yTestList.append(nameClass)
        xTestList.append(X[0])

Classes = ['Relaxado', 'Pedra', 'Papel', 'Tesoura']
for i in Classes:
    getTestFeatures(i, xTestList, yTestList, maximus)
    if i == "Relaxado":
        threshold = max(maximus)

# X_train, X_test, y_train, y_test = train_test_split(xTestList, yTestList, test_size = 0.10, shuffle = True)
# print(y_test)

clf = svm.SVC()
clf.fit(xTestList, yTestList)
#clf.fit(X_train, y_train)

# predicted = clf.predict(X_test)

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()

# The macAddress variable on Windows can be "XX:XX:XX:XX:XX:XX" or "COMX"
# while on Mac OS can be "/dev/tty.BITalino-XX-XX-DevB" for devices ending with the last 4 digits of the MAC address or "/dev/tty.BITalino-DevB" for the remaining
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
    for i in df:
        if i > threshold:
            return False
    else:
        return True

turnOff = False
window = np.zeros(1000)

while True:
    # Read samples
    sample = device.read(nSamples)
    window = np.roll(window, -nSamples)
    window[(1000 - nSamples):] = sample[:, 5]
    window2 = abs(window - window.mean())
    #if not is_relaxed(window2, threshold):
    features = tsfel.time_series_features_extractor(cfg, window2)
    predicted = clf.predict(features)
    print(predicted)

    # if predicted != "Relaxado":
    #     features = tsfel.time_series_features_extractor(cfg, window2)
    #     predicted = clf.predict(features)

    if turnOff:
        break

# Stop acquisition
device.stop()

# Close connection
device.close()