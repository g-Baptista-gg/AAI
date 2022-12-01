import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import seaborn as sns
import novainstrumentation as ni
plt.rcParams["figure.autolayout"]=True
import scipy.signal as signal


files_relaxado = os.listdir(os.getcwd()+'/Relaxado')[1::2]
files_pedra = os.listdir(os.getcwd()+'/Pedra')[1::2]
files_papel = os.listdir(os.getcwd()+'/Papel')[1::2]
files_tesoura = os.listdir(os.getcwd()+'/Tesoura')[1::2]

rel = [pd.read_csv(os.getcwd()+'/Relaxado/'+i, header = 3, delimiter = '\t')for i in files_relaxado]
pedra = [pd.read_csv(os.getcwd()+'/Pedra/'+i, header = 3, delimiter = '\t')for i in files_pedra]
papel = [pd.read_csv(os.getcwd()+'/Papel/'+i, header = 3, delimiter = '\t')for i in files_papel]
tesoura = [pd.read_csv(os.getcwd()+'/Tesoura/'+i, header = 3, delimiter = '\t')for i in files_tesoura]

fig, ax = plt.subplots(2, 2,sharex=True,sharey=True)
fig.suptitle('Signal\'s FFTs')
fig1, ax1 = plt.subplots(2, 2)


lw=1
window_size=81

feature_csv = open('features.csv','w')
writer = csv.writer(feature_csv,lineterminator='\n')
writer.writerow(['Feature','Max Freq','Desv Pad', 'Max', 'Freq Med'])

ax[0, 0].set_title("Relaxado")
ax1[0, 0].set_title("Relaxado")
print('-----------RELAXADO-----------')
for i in rel:
    arraytest = np.array(i.iloc[:, 5])
    arraytest = arraytest - arraytest.mean()
    fft1 = abs(fft(arraytest))
    freq = fftfreq(len(arraytest), 0.001)
    freq = ni.smooth(freq)
    ax1[0, 0].plot(arraytest/arraytest.std(),alpha=0.5,linewidth=lw)
    ax[0, 0].plot(freq[freq>0],signal.medfilt(fft1[freq>=0]/fft1.std(),kernel_size=window_size),alpha=0.8,linewidth=lw)
    print('STD:\t',signal.medfilt(fft1[freq>=0]/fft1.std()).std())
    writer.writerow(['rel',freq[fft1.argmax()],signal.medfilt(fft1[freq>=0]/fft1.std()).std(), arraytest.max()/arraytest.std(), fft1[freq>=0].mean()])

print('-----------PEDRA-----------')
ax[0, 1].set_title("Pedra")
ax1[0, 1].set_title("Pedra")
for i in pedra:
    arraytest = np.array(i.iloc[:, 5])
    arraytest = arraytest - arraytest.mean()
    fft1 = abs(fft(arraytest))
    freq = fftfreq(len(arraytest), 0.001)
    freq = ni.smooth(freq)
    ax1[0, 1].plot(arraytest/arraytest.std(),alpha=0.5,linewidth=lw)
    ax[0, 1].plot(freq[freq>=0], signal.medfilt(fft1[freq>=0]/fft1.std(),kernel_size=window_size),alpha=0.8,linewidth=lw)
    print('STD:\t',signal.medfilt(fft1[freq>=0]/fft1.std()).std())
    writer.writerow(['pedra',freq[fft1.argmax()],signal.medfilt(fft1[freq>=0]/fft1.std()).std(), arraytest.max()/arraytest.std(), fft1[freq>=0].mean()])
 

print('-----------PAPEL-----------') 
ax[1, 0].set_title("Papel")
ax1[1, 0].set_title("Papel")
for i in papel:
    arraytest = np.array(i.iloc[:, 5])
    arraytest = arraytest - arraytest.mean()
    fft1 = abs(fft(arraytest))
    freq = fftfreq(len(arraytest), 0.001)
    freq = ni.smooth(freq)
    ax1[1, 0].plot(arraytest/arraytest.std(),alpha=0.5,linewidth=lw)
    ax[1, 0].plot(freq[freq>=0], signal.medfilt(fft1[freq>=0]/fft1.std(),kernel_size=window_size),alpha=0.8,linewidth=lw)
    print('STD:\t',signal.medfilt(fft1[freq>=0]/fft1.std()).std())
    writer.writerow(['papel',freq[fft1.argmax()],signal.medfilt(fft1[freq>=0]/fft1.std()).std(), arraytest.max()/arraytest.std(), fft1[freq>=0].mean()])

print('-----------TESOURA-----------')
ax[1, 1].set_title("Tesoura")
ax1[1, 1].set_title("Tesoura")
for i in tesoura:
    arraytest = np.array(i.iloc[:, 5])
    arraytest = arraytest - arraytest.mean()
    fft1 = abs(fft(arraytest))
    freq = fftfreq(len(arraytest), 0.001)
    freq = ni.smooth(freq)
    ax1[1, 1].plot(arraytest/arraytest.std(),alpha=0.5,linewidth=lw)
    ax[1, 1].plot(freq[freq>=0], signal.medfilt(fft1[freq>=0]/fft1.std(),kernel_size=window_size),alpha=0.8,linewidth=lw)
    print('STD:\t',signal.medfilt(fft1[freq>=0]/fft1.std()).std())
    writer.writerow(['tes',freq[fft1.argmax()],signal.medfilt(fft1[freq>=0]/fft1.std()).std(), arraytest.max()/arraytest.std(), fft1[freq>=0].mean()])
feature_csv.close()

plotdf = pd.read_csv('features.csv')
#sns.relplot(data = plotdf, x = 'Max Freq', y = 'Desv Pad', hue = 'Feature')
#sns.relplot(data = plotdf, x = 'Max', y = 'Desv Pad', hue = 'Feature')
#sns.relplot(data = plotdf, x = 'Max Freq', y = 'Max', hue = 'Feature')

sns.pairplot(data = plotdf, hue = 'Feature')

plt.show()


