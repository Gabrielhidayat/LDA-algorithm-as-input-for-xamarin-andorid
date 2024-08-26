# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:51:35 2020

@author: hiday
"""

import os
import fnmatch
import pandas as pd
from numpy import mean
import seaborn as sns

'url1 adalah nama file tempat datamentah'
'dg lostdir dan fnmatch, mensortir file khusus untuk csv'
url1 = 'data plot LDA/data februari 2020/semua'
List1 = os.listdir(url1)
List1 = [i for i in List1 if fnmatch.fnmatch(i, '*.csv')]

url2 = 'data plot LDA/data Maret 2020/semua'
List2 = os.listdir(url2)
List2 = [i for i in List2 if fnmatch.fnmatch(i, '*.csv')]

'mendefinisikan fungsi rata2'
def fc(x):
    return mean(x)
'hasil adalah tempat unt uk menaruh semua hasil perhitungan'
hasil1 = None
hasil2 = None


for i, item in enumerate(List1):
    temp = pd.read_csv(f'{url1}/{item}')
    temp = temp.iloc[:, 1:17]
    'mengembalikan pada fungsi fc(x)'
    temp = temp.apply(fc)
    
    x1=item.split('_')
    temp['label'] = x1[1]
    temp['tanggal']= x1[2]
    
    temp = pd.DataFrame(data = temp)
    temp = temp.transpose()
    
    if i == 0:
        hasil1 = temp
        
    else:
        hasil1 = pd.concat([
            hasil1,
            temp,
        ], axis=0)
hasil1.to_csv(f'hasil bulan 1 (februari).csv', index=False)

for i, item in enumerate(List2):
    temp2 = pd.read_csv(f'{url2}/{item}')
    temp2 = temp2.iloc[:, 1:17]
    'mengembalikan pada fungsi fc(x)'
    temp2 = temp2.apply(fc)
    
    x2=item.split('_')
    temp2['label'] = x2[1]
    temp2['tanggal']= x2[2]
    
    temp2 = pd.DataFrame(data = temp2)
    temp2 = temp2.transpose()
    
    if i == 0:
        hasil2 = temp2
        
    else:
        hasil2 = pd.concat([
            hasil2,
            temp2,
        ], axis=0)
hasil2.to_csv(f'hasil bulan 2 (maret).csv', index=False)

df1= pd.read_csv('/hasil bulan 1 (februari).csv')
df2= pd.read_csv('/hasil bulan 2 (maret).csv')

X1=df1.values[:,0:16].astype(float)
y1=df1.values[:,16]

X2=df2.values[:,0:16].astype(float)
y2=df2.values[:,16]

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
model = Pipeline([
        ('scale',StandardScaler()),
        ('lda', LinearDiscriminantAnalysis()),
        ])
    
hasil = cross_val_score(model, X=X1,y=y1, cv=kfold, n_jobs=-1)

print(f'nilai rata2 :{hasil.mean()}')
print(f'standard deviation :{hasil.std()}')


'''
X_train,X_test,y_train,y_test= train_test_split(X,y,stratify=y, test_size=0.2, random_state=1)
'''


X_train=X1
y_train=y1
X_test=X2
y_test=y2

model.fit(X_train, y_train)
model.score(X_train,y_train)
model.score(X_test,y_test)

model.score(X_test,y_train)

model['lda'].coef_
model['lda'].intercept_

model['scale'].mean_

model['scale'].var_

'scalling'
model['scale'].fit(X_train)
xtrainscaled = model['scale'].transform(X_train)
xtestscaled = model['scale'].transform(X_test)
'transform to LD value'
ldtrain = model['lda'].transform(xtrainscaled)
ldtest = model['lda'].transform(xtestscaled)
'variansi LD'
ldname = [f'LD{i + 1}' for i in range(ldtrain.shape[1])]
tot = sum(model['lda'].explained_variance_ratio_)
ldvar = [round((i / tot) * 100, 1) for i in sorted(model['lda'].explained_variance_ratio_, reverse=True)]
ldname = [f'{a} ({b}%)' for a, b in zip(ldname, ldvar)]
print(ldname)
'dataframe for plotting'
ldtraindf = pd.DataFrame(ldtrain, columns=ldname)
ldtraindf = pd.concat([ldtraindf, pd.DataFrame(y_train, columns=['kelas'])], axis=1)
ldtraindf['case'] = 'training'

ldtestdf = pd.DataFrame(ldtest, columns=ldname)
ldtestdf = pd.concat([ldtestdf, pd.DataFrame(y_test, columns=['kelas'])], axis=1)
ldtestdf['case'] = 'testing'

ldDf = pd.concat([ldtraindf, ldtestdf], axis=0)
ldDf = ldDf.reset_index(drop=True)

'matplotlib notebook'
sns.scatterplot(x=ldname[0], y=ldname[1], hue='kelas', size='case', style='kelas', data=ldDf)

pred = model.predict(X2)
hasil_label=classification_report(y_test,pred)

df2.label.unique()
























