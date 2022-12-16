import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import tsfel
from bitalino import BITalino

# The macAddress variable on Windows can be "XX:XX:XX:XX:XX:XX" or "COMX"
# while on Mac OS can be "/dev/tty.BITalino-XX-XX-DevB" for devices ending with the last 4 digits of the MAC address or "/dev/tty.BITalino-DevB" for the remaining
macAddress = "20:18:05:28:73:28"

batteryThreshold = 30
acqChannels = [1]
samplingRate = 1000
nSamples = 50

# Connect to BITalino
device = BITalino(macAddress)

# Set battery threshold
device.battery(batteryThreshold)

# Read BITalino version
print(device.version())

# Start Acquisition
device.start(samplingRate, acqChannels)

files_relaxado = os.listdir(os.getcwd() + '/Relaxado')[1::2]
files_pedra = os.listdir(os.getcwd() + '/Pedra')[1::2]
files_papel = os.listdir(os.getcwd() + '/Papel')[1::2]
files_tesoura = os.listdir(os.getcwd() + '/Tesoura')[1::2]

rel = [pd.read_csv(os.getcwd() + '/Relaxado/' + i, header = 3, delimiter = '\t')for i in files_relaxado]
pedra = [pd.read_csv(os.getcwd() + '/Pedra/' + i, header = 3, delimiter = '\t')for i in files_pedra]
papel = [pd.read_csv(os.getcwd() + '/Papel/' + i, header = 3, delimiter = '\t')for i in files_papel]
tesoura = [pd.read_csv(os.getcwd() + '/Tesoura/' + i, header = 3, delimiter = '\t')for i in files_tesoura]

feature_csv = open('features2.csv','w')
writer = csv.writer(feature_csv, lineterminator = '\n')

df = pd.read_csv('Papel/opensignals_201805286295_2022-11-29_15-03-12.txt', skiprows = (3), header = None, delimiter = '\t')
df = np.array(df[5])

window = np.zeros(1000)

i = 0

def is_relaxed(df, threshold):
    for i in df:
        if abs(i) > threshold:
            return False
    else:
        return True

while True:
    # Read samples
    sample = device.read(nSamples) - 512
    window = np.roll(window, -50)
    window[950:] = sample[:, 5]
    print(sample[:, 5])
    print(is_relaxed(window, 30))
    input(i)

cfg = tsfel.get_features_by_domain()

#X = tsfel.time_series_features_extractor(cfg, df)
#X = X.head()
#Y = ["Feature"]
#for i in X:
#    Y.append(i)
#print(type(Y))
#writer.writerow(Y)

fig, ax = plt.subplots(2, sharex=True)

maximus = []

for i in rel:
    df = np.array(i.iloc[:, 5])
    df = abs(df - df.mean())
    maximus.append(max(df))
    #ax[0].plot(df)

threshold = max(maximus)
#print(threshold)

def cut2(df, threshold):
    for i in range(len(df)):
        if df[i] > threshold:
            return df[i:]

def cut(df, threshold):
    for i in range(len(df)):
        if df[i] > threshold:
            return df[(i+500):]

for i in pedra:
    df = np.array(i.iloc[:, 5])
    df = df - df.mean()
    X = tsfel.time_series_features_extractor(cfg, cut(df, threshold))
    #X = np.array(X)
    X = X.values.tolist()
    Y = ["pedra"]
    for i in X[0]:
        Y.append(i)
    writer.writerow(Y)
    #ax[0].plot(cut2(df, threshold))
    #ax[1].plot(cut(df, threshold))

for i in papel:
    df = np.array(i.iloc[:, 5])
    df = df - df.mean()
    X = tsfel.time_series_features_extractor(cfg, cut(df, threshold))
    X = np.array(X)
    X = X.tolist()
    Y = ["papel"]
    for i in X[0]:
        Y.append(i)
    writer.writerow(Y)
    #ax[0].plot(cut2(df, threshold))
    #ax[1].plot(cut(df, threshold))

for i in tesoura:
    df = np.array(i.iloc[:, 5])
    df = df - df.mean()
    X = tsfel.time_series_features_extractor(cfg, cut(df, threshold))
    X = np.array(X)
    X = X.tolist()
    Y = ["tesoura"]
    for i in X[0]:
        Y.append(i)
    writer.writerow(Y)
    #ax[0].plot(cut2(df, threshold))
    #ax[1].plot(cut(df, threshold))

feature_csv.close()

#plt.show()

# Stop acquisition
device.stop()

# Close connection
device.close()