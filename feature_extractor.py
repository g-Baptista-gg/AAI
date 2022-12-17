import tsfel
import pandas as pd
import numpy as np
import os
import csv

files_relaxado = os.listdir(os.getcwd() + '/Relaxado')[1::2]
files_pedra = os.listdir(os.getcwd() + '/Pedra')[1::2]
files_papel = os.listdir(os.getcwd() + '/Papel')[1::2]
files_tesoura = os.listdir(os.getcwd() + '/Tesoura')[1::2]

rel = [pd.read_csv(os.getcwd() + '/Relaxado/' + i, header = 3, delimiter = '\t')for i in files_relaxado]
pedra = [pd.read_csv(os.getcwd() + '/Pedra/' + i, header = 3, delimiter = '\t')for i in files_pedra]
papel = [pd.read_csv(os.getcwd() + '/Papel/' + i, header = 3, delimiter = '\t')for i in files_papel]
tesoura = [pd.read_csv(os.getcwd() + '/Tesoura/' + i, header = 3, delimiter = '\t')for i in files_tesoura]

feature_csv = open('features1.csv','w')
writer = csv.writer(feature_csv, lineterminator = '\n')

# load dataset
df = pd.read_csv('Papel/opensignals_201805286295_2022-11-29_15-03-12.txt', skiprows = (3), header = None, delimiter = '\t')
df = np.array(df[5])

# Retrieves a pre-defined feature configuration file to extract all available features
cfg = tsfel.get_features_by_domain()

X = tsfel.time_series_features_extractor(cfg, df)
X = X.head()
Y = ["Feature"]
for i in X:
    Y.append(i)
writer.writerow(Y)

features_to_save=[]
for i in range(len(Y)):
    if Y[i] in ["0_Sum absolute diff","0_Median absolute diff","0_Signal distance","0_FFT mean coefficient_177","0_FFT mean coefficient_194","0_Wavelet variance_0","0_Wavelet standard deviation_0","0_Wavelet energy_0","0_FFT mean coefficient_58","0_LPCC_11"]:
        features_to_save.append(i)

for i in rel:
    df = np.array(i.iloc[:, 5])
    df = df - df.mean()
    X = tsfel.time_series_features_extractor(cfg, df)
    #X = np.array(X)
    X = X.values.tolist()
    Y = ["relaxado"]
    for i in X[0]:
        Y.append(i)
    writer.writerow(Y)
    
    
for i in pedra:
    df = np.array(i.iloc[:, 5])
    df = df - df.mean()
    X = tsfel.time_series_features_extractor(cfg, df)
    X = np.array(X)
    X = X.tolist()
    Y = ["pedra"]
    for i in X[0]:
        Y.append(i)
    print(type(Y))
    writer.writerow(Y)
 

for i in papel:
    df = np.array(i.iloc[:, 5])
    df = df - df.mean()
    X = tsfel.time_series_features_extractor(cfg, df)
    X = np.array(X)
    X = X.tolist()
    Y = ["papel"]
    for i in X[0]:
        Y.append(i)
    writer.writerow(Y)


for i in tesoura:
    df = np.array(i.iloc[:, 5])
    df = df - df.mean()
    X = tsfel.time_series_features_extractor(cfg, df)
    X = np.array(X)
    X = X.tolist()
    Y = ["tesoura"]
    for i in X[0]:
        Y.append(i)
    writer.writerow(Y)
    
feature_csv.close()
