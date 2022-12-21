import numpy as np
from sklearn import ensemble
from bitalino import BITalino
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import sys
import pickle
from featureExtrationEMG import featureExtraction

filename = 'trained_classifier.sav'
filenameThreshold = 'threshold.sav'
clf = pickle.load(open(filename, 'rb'))
threshold = pickle.load(open(filenameThreshold, 'rb'))

def signalParts(df, threshold):
    for i in range(len(df)):
        if df[i] > (1.1 * threshold):
            return df[i:(i + 500)], df[(i + 500):(i + 2000)]
    else:
        return df[:500],df[500:2000]

macAddress = "20:18:05:28:73:28"

# Connect to BITalino
device = BITalino(macAddress)

# Read BITalino version
print(device.version())

def is_relaxed(df, threshold):
    for i in abs(df):
        if i > (1.1 * threshold):
            return False
    else:
        return True

def predInt(predicted):
    if (predicted[0] == 'Relaxado'):
        return 0
    elif (predicted[0] == 'Pedra'):
        return 1
    elif (predicted[0] == 'Papel'):
        return 2
    else:
        return 3

def classify():

    acqChannels = [0]
    samplingRate = 1000
    nSamples = 3000

    device.start(samplingRate, acqChannels)

    # Read samples
    sample = device.read(nSamples)[:, 5]
    signal = sample - sample.mean()

    sigAc,sigPos = signalParts(signal, threshold)
    features = featureExtraction(sigAc, sigPos)
    predicted = clf.predict(features.reshape(1, -1))

    device.stop()
    print(predicted, file = sys.stdout)

    return predInt(predicted)

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def sessions():
    return render_template('index.html')

@socketio.on('acquire')
def start_acquisition(json):
    emit('serverResponse', {'data': classify()})

if __name__ == '__main__':
    socketio.run(app, debug = True)

# Close connection
#device.close()