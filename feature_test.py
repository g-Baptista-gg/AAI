import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import datasets, svm,metrics
from sklearn.model_selection import train_test_split
import tsfel

cfg = tsfel.get_features_by_domain()
def cut(df, threshold):
    for i in range(len(df)):
        if df[i] > threshold:
            return df[i:(i + 500)]
    else: return df

files_relaxado = os.listdir(os.getcwd() + '/Relaxado')[1::2]
files_pedra = os.listdir(os.getcwd() + '/Pedra')[1::2]
files_papel = os.listdir(os.getcwd() + '/Papel')[1::2]
files_tesoura = os.listdir(os.getcwd() + '/Tesoura')[1::2]

rel = [pd.read_csv(os.getcwd() + '/Relaxado/' + i, header = 3, delimiter = '\t')for i in files_relaxado]
pedra = [pd.read_csv(os.getcwd() + '/Pedra/' + i, header = 3, delimiter = '\t')for i in files_pedra]
papel = [pd.read_csv(os.getcwd() + '/Papel/' + i, header = 3, delimiter = '\t')for i in files_papel]
tesoura = [pd.read_csv(os.getcwd() + '/Tesoura/' + i, header = 3, delimiter = '\t')for i in files_tesoura]
k=0
color=['black','red','green','blue']

sig_class=[]
sig_feature=[]
class_name=['Relaxado','Pedra','Papel','Tesoura']
for j in [rel,pedra,papel,tesoura]:
    #fig,ax=plt.subplots()
    for i in j:
        i=np.array(i)[:,5]
        i-=i.mean()
        i=abs(i)
        i=cut(i,25)
        i/=i.std()
        sig_feature.append(np.array(tsfel.time_series_features_extractor(cfg,i))[0])
        sig_class.append(class_name[k])
    k+=1

print(sig_feature)

X_train, X_test, y_train, y_test = train_test_split(sig_feature, sig_class, test_size = 0.33, shuffle = True)
clf = svm.SVC()


#print(y_train)
#clf.fit(X_train,y_train)

#predicted = clf.predict(X_test)

#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)



plt.show()