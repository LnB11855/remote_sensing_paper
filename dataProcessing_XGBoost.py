import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import gc
import xgboost as xgb
from xgboost import plot_importance
import numpy as np

from sklearn.preprocessing import StandardScaler,normalize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation,GRU
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
seq_length=6
plt.close('all')
coloumns = ["Date", "Quadrat", "Blue", "Green", "Red", "NIR", "Rotation", "NDVI", "Inc"]
rawData = pd.read_csv("ALL_Bands_2016.csv")
rawData['Date'] = pd.to_datetime(rawData.Date, format='%m/%d/%Y')
rawData['Date']=rawData['Date'].dt.date
cleanedDataOne = rawData[rawData['Date'] == pd.datetime(2016, 8, 5).date()]
cleanedDataOne = cleanedDataOne.rename(columns={'Inc_Aug_5': 'Inc'})
cleanedDataOne = cleanedDataOne[coloumns]

cleanedDataTwo = rawData[rawData['Date'] == pd.datetime(2016, 8, 21).date()]
cleanedDataTwo = cleanedDataTwo.rename(columns={'Inc_Aug_22': 'Inc'})
cleanedDataTwo = cleanedDataTwo[coloumns]

cleanedDataThree = rawData[rawData['Date'] == pd.datetime(2016, 8, 31).date()]
cleanedDataThree = cleanedDataThree.rename(columns={'Inc_Aug_29': 'Inc'})
cleanedDataThree = cleanedDataThree[coloumns]

cleanedDataFour = rawData[rawData['Date'] == pd.datetime(2016, 7, 5).date()]
cleanedDataFour = cleanedDataFour.rename(columns={'Inc_Aug_29': 'Inc'})
cleanedDataFour = cleanedDataFour[coloumns]
cleanedDataFour['Inc'] =0

cleanedDataFive = rawData[rawData['Date'] == pd.datetime(2016, 7, 9).date()]
cleanedDataFive = cleanedDataFive.rename(columns={'Inc_Aug_29': 'Inc'})
cleanedDataFive = cleanedDataFive[coloumns]
cleanedDataFive['Inc'] =0

cleanedDataSix = rawData[rawData['Date'] == pd.datetime(2016, 7, 20).date()]
cleanedDataSix = cleanedDataSix.rename(columns={'Inc_Aug_29': 'Inc'})
cleanedDataSix = cleanedDataSix[coloumns]
cleanedDataSix['Inc'] =0

cleanedData = pd.concat([cleanedDataOne,cleanedDataTwo,cleanedDataThree,cleanedDataFour,cleanedDataFive,cleanedDataSix])
#cleanedData = pd.concat([cleanedDataOne,cleanedDataTwo,cleanedDataThree])
# # fig, ax = plt.subplots(figsize=(10, 10))
# plot = sns.jointplot(x=cleanedData['SDS'], y=cleanedData['Inc_Sep_06'], kind='kde', color='blueviolet')
# plt.show()
df = cleanedData.groupby('Quadrat').cumcount()
print('grouping by Date,Quadrat, combines Blue_Mean')
gp = cleanedData[['Date', 'Quadrat', 'Blue']].groupby(['Date', 'Quadrat'])[['Blue']].mean().reset_index().rename(
    index=str, columns={'Blue': 'Blue_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Blue_std')
gp = cleanedData[['Date', 'Quadrat', 'Blue']].groupby(['Date', 'Quadrat'])[['Blue']].std().reset_index().rename(
    index=str, columns={'Blue': 'Blue_std'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Green_Mean')
gp = cleanedData[['Date', 'Quadrat', 'Green']].groupby(['Date', 'Quadrat'])[['Green']].mean().reset_index().rename(
    index=str, columns={'Green': 'Green_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Green_std')
gp = cleanedData[['Date', 'Quadrat', 'Green']].groupby(['Date', 'Quadrat'])[['Green']].std().reset_index().rename(
    index=str, columns={'Green': 'Green_std'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Red_Mean')
gp = cleanedData[['Date', 'Quadrat', 'Red']].groupby(['Date', 'Quadrat'])[['Red']].mean().reset_index().rename(
    index=str, columns={'Red': 'Red_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Red_std')
gp = cleanedData[['Date', 'Quadrat', 'Red']].groupby(['Date', 'Quadrat'])[['Red']].std().reset_index().rename(index=str,
                                                                                                              columns={
                                                                                                                  'Red': 'Red_std'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NIR_Mean')
gp = cleanedData[['Date', 'Quadrat', 'NIR']].groupby(['Date', 'Quadrat'])[['NIR']].mean().reset_index().rename(
    index=str, columns={'NIR': 'NIR_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NIR_std')
gp = cleanedData[['Date', 'Quadrat', 'NIR']].groupby(['Date', 'Quadrat'])[['NIR']].std().reset_index().rename(index=str,
                                                                                                              columns={
                                                                                                                  'NIR': 'NIR_std'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NDVI_Mean')
gp = cleanedData[['Date', 'Quadrat', 'NDVI']].groupby(['Date', 'Quadrat'])[['NDVI']].mean().reset_index().rename(
    index=str, columns={'NDVI': 'NDVI_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NDVI_std')
gp = cleanedData[['Date', 'Quadrat', 'NDVI']].groupby(['Date', 'Quadrat'])[['NDVI']].std().reset_index().rename(
    index=str, columns={'NDVI': 'NDVI_std'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()

# cleanedData.loc[cleanedData['Rotation'] == "S4", 'Rotation'] = 1
# cleanedData.loc[cleanedData['Rotation'] == "S3", 'Rotation'] = 0
# cleanedData.loc[cleanedData['Rotation'] == "S2", 'Rotation'] = -1
one_hot = pd.get_dummies(cleanedData['Rotation'])
# Drop column B as it is now encoded
cleanedData= cleanedData.drop('Rotation',axis = 1)
# Join the encoded df
cleanedData = cleanedData.join(one_hot)

cleanedData['Inc'] = pd.cut(cleanedData['Inc'], [0,5,101],labels=np.arange(2), right=False)


dropCol = ['Blue', 'Green', 'Red', 'NIR', 'NDVI']
cleanedDataDel=cleanedData.drop(columns=dropCol).drop_duplicates().reset_index(drop=True)
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 7, 5).date(),'Date']=0
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 7, 9).date(),'Date']=1
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 7, 20).date(),'Date']=2
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 8, 5).date(),'Date']=3
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 8, 21).date(),'Date']=4
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 8, 31).date(),'Date']=5

scaler_cols=['Blue_Mean','Red_Mean','Green_Mean','NIR_Mean','NDVI_Mean','Blue_std','Red_std','Green_std','NIR_std','NDVI_std']
cleanedDataDel[scaler_cols] = StandardScaler().fit_transform(cleanedDataDel[scaler_cols])
cleanedDataDel=cleanedDataDel.sort_values(by=['Date'])

cleanedDataDel=cleanedDataDel[cleanedDataDel.Date>=seq_length-1]
cleanedDataDel['year']=0
Q_train=[]  #define train dataset
Q_test=[]   #defin test dataset
trainData = cleanedDataDel[cleanedDataDel['Quadrat'] .isin(Q_train)].rename(columns={"Quadrat": "id"})
testData = cleanedDataDel[cleanedDataDel['Quadrat'] .isin(Q_test)0].rename(columns={"Quadrat": "id"})



# dropCol = ['Date', 'Inc', 'Blue', 'Green', 'Red', 'NIR', 'NDVI', 'Quadrat']
dropCol = ['Date', 'Inc', 'id']
x1 = trainData.drop(columns=dropCol)
y1 = trainData['Inc']
x2 = testData.drop(columns=dropCol)
y2 = testData['Inc']
print(x1.shape)
print(x2.shape)


plt.close('all')
coloumns = ["Date", "Quadrat", "Blue", "Green", "Red", "NIR", "Rotation", "NDVI", "Inc"]
rawData = pd.read_csv("ALL_Bands_2017.csv")
rawData['Date'] = pd.to_datetime(rawData.Date, format='%m/%d/%Y')
rawData['Date']=rawData['Date'].dt.date
cleanedDataOne = rawData[rawData['Date'] == pd.datetime(2017, 7, 31).date()]
cleanedDataOne = cleanedDataOne.rename(columns={'Inc_Aug_31': 'Inc'})
cleanedDataOne = cleanedDataOne[coloumns]
cleanedDataOne['Inc'] =0

cleanedDataTwo = rawData[rawData['Date'] == pd.datetime(2017, 8, 18).date()]
cleanedDataTwo = cleanedDataTwo.rename(columns={'Inc_Aug_17': 'Inc'})
cleanedDataTwo = cleanedDataTwo[coloumns]

cleanedDataThree = rawData[rawData['Date'] == pd.datetime(2017, 8, 23).date()]
cleanedDataThree = cleanedDataThree.rename(columns={'Inc_Aug_24': 'Inc'})
cleanedDataThree = cleanedDataThree[coloumns]

cleanedDataFour = rawData[rawData['Date'] == pd.datetime(2017, 7, 5).date()]
cleanedDataFour = cleanedDataFour.rename(columns={'Inc_Aug_31': 'Inc'})
cleanedDataFour = cleanedDataFour[coloumns]
cleanedDataFour['Inc'] =0

cleanedDataFive = rawData[rawData['Date'] == pd.datetime(2017, 7, 9).date()]
cleanedDataFive = cleanedDataFive.rename(columns={'Inc_Aug_31': 'Inc'})
cleanedDataFive = cleanedDataFive[coloumns]
cleanedDataFive['Inc'] =0

cleanedDataSix = rawData[rawData['Date'] == pd.datetime(2017, 7, 20).date()]
cleanedDataSix = cleanedDataSix.rename(columns={'Inc_Aug_31': 'Inc'})
cleanedDataSix = cleanedDataSix[coloumns]
cleanedDataSix['Inc'] =0

cleanedData = pd.concat([cleanedDataOne,cleanedDataTwo,cleanedDataThree,cleanedDataFour,cleanedDataFive,cleanedDataSix])
#cleanedData = pd.concat([cleanedDataOne,cleanedDataTwo,cleanedDataThree])
# # fig, ax = plt.subplots(figsize=(10, 10))
# plot = sns.jointplot(x=cleanedData['SDS'], y=cleanedData['Inc_Sep_06'], kind='kde', color='blueviolet')
# plt.show()
df = cleanedData.groupby('Quadrat').cumcount()
print('grouping by Date,Quadrat, combines Blue_Mean')
gp = cleanedData[['Date', 'Quadrat', 'Blue']].groupby(['Date', 'Quadrat'])[['Blue']].mean().reset_index().rename(
    index=str, columns={'Blue': 'Blue_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Blue_std')
gp = cleanedData[['Date', 'Quadrat', 'Blue']].groupby(['Date', 'Quadrat'])[['Blue']].std().reset_index().rename(
    index=str, columns={'Blue': 'Blue_std'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Green_Mean')
gp = cleanedData[['Date', 'Quadrat', 'Green']].groupby(['Date', 'Quadrat'])[['Green']].mean().reset_index().rename(
    index=str, columns={'Green': 'Green_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Green_std')
gp = cleanedData[['Date', 'Quadrat', 'Green']].groupby(['Date', 'Quadrat'])[['Green']].std().reset_index().rename(
    index=str, columns={'Green': 'Green_std'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Red_Mean')
gp = cleanedData[['Date', 'Quadrat', 'Red']].groupby(['Date', 'Quadrat'])[['Red']].mean().reset_index().rename(
    index=str, columns={'Red': 'Red_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines Red_std')
gp = cleanedData[['Date', 'Quadrat', 'Red']].groupby(['Date', 'Quadrat'])[['Red']].std().reset_index().rename(index=str,
                                                                                                              columns={
                                                                                                                  'Red': 'Red_std'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NIR_Mean')
gp = cleanedData[['Date', 'Quadrat', 'NIR']].groupby(['Date', 'Quadrat'])[['NIR']].mean().reset_index().rename(
    index=str, columns={'NIR': 'NIR_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NIR_std')
gp = cleanedData[['Date', 'Quadrat', 'NIR']].groupby(['Date', 'Quadrat'])[['NIR']].std().reset_index().rename(index=str,
                                                                                                              columns={
                                                                                                                  'NIR': 'NIR_std'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NDVI_Mean')
gp = cleanedData[['Date', 'Quadrat', 'NDVI']].groupby(['Date', 'Quadrat'])[['NDVI']].mean().reset_index().rename(
    index=str, columns={'NDVI': 'NDVI_Mean'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()
print('grouping by Date,Quadrat, combines NDVI_std')
gp = cleanedData[['Date', 'Quadrat', 'NDVI']].groupby(['Date', 'Quadrat'])[['NDVI']].std().reset_index().rename(
    index=str, columns={'NDVI': 'NDVI_std'})
cleanedData = cleanedData.merge(gp, on=['Date', 'Quadrat'], how='left')
del gp
gc.collect()

one_hot = pd.get_dummies(cleanedData['Rotation'])
# Drop column B as it is now encoded
cleanedData= cleanedData.drop('Rotation',axis = 1)
# Join the encoded df
cleanedData = cleanedData.join(one_hot)

cleanedData['Inc'] = pd.cut(cleanedData['Inc'], [0,5,101],labels=np.arange(2), right=False)

dropCol = ['Blue', 'Green', 'Red', 'NIR', 'NDVI']
cleanedDataDel=cleanedData.drop(columns=dropCol).drop_duplicates().reset_index(drop=True)
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 7, 5).date(),'Date']=0
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 7, 9).date(),'Date']=1
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 7, 20).date(),'Date']=2
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 7, 31).date(),'Date']=3
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 8, 18).date(),'Date']=4
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 8, 23).date(),'Date']=5
cleanedDataDel['year']=1
scaler_cols=['Blue_Mean','Red_Mean','Green_Mean','NIR_Mean','NDVI_Mean','Blue_std','Red_std','Green_std','NIR_std','NDVI_std']
cleanedDataDel[scaler_cols] = StandardScaler().fit_transform(cleanedDataDel[scaler_cols])
cleanedDataDel=cleanedDataDel.sort_values(by=['Date','Quadrat'])

if seq_length<6:
    cleanedDataDel=cleanedDataDel[cleanedDataDel.Date>=4]
else:
    cleanedDataDel = cleanedDataDel[cleanedDataDel.Date >=5]
trainData = cleanedDataDel[cleanedDataDel['Quadrat'] .isin(Q_train)].rename(columns={"Quadrat": "id"})
testData = cleanedDataDel[cleanedDataDel['Quadrat'] .isin(Q_test)].rename(columns={"Quadrat": "id"})



# dropCol = ['Date', 'Inc', 'Blue', 'Green', 'Red', 'NIR', 'NDVI', 'Quadrat']
dropCol = ['Date', 'Inc', 'id']
x3 = trainData.drop(columns=dropCol)
y3 = trainData['Inc']
x4 = testData.drop(columns=dropCol)
y4 = testData['Inc']
print(x3.shape)
print(x4.shape)

X_train=pd.concat((x1,x3))
y_train=pd.concat((y1,y3))
X_test=pd.concat((x2,x4))
y_test=pd.concat((y2,y4))
with open('dataXGBoost'+str(seq_length)+'.pickle', 'wb') as f:
    pickle.dump([X_train, X_test, y_train,y_test], f)
