import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import keras
import gc
import xgboost as xgb
import pickle
from xgboost import plot_importance
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation,GRU, Bidirectional
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
seq_length=4
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
uniqueQuadrat1=cleanedData['Quadrat'].unique()
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

def gen_sequence(id_df, seq_length, seq_cols):
    # df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    # id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)
def gen_label(id_df, seq_length, seq_cols,label):
    # df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    # id_df=df_zeros.append(id_df,ignore_index=True)

    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label=[]
    for stop in range(seq_length-1, num_elements):
        y_label.append(id_df[label].values[stop])
    return np.array(y_label)

dropCol = ['Blue', 'Green', 'Red', 'NIR', 'NDVI']
cleanedDataDel=cleanedData.drop(columns=dropCol).drop_duplicates().reset_index(drop=True)
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 7, 5).date(),'Date']=0
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 7, 9).date(),'Date']=1
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 7, 20).date(),'Date']=2
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 8, 5).date(),'Date']=3
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 8, 21).date(),'Date']=4
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2016, 8, 31).date(),'Date']=5

one_hot_2 = pd.get_dummies(cleanedDataDel['Date'],prefix='col')
# Drop column B as it is now encoded

cleanedDataDel = cleanedDataDel.join(one_hot_2)

scaler_cols=['Blue_Mean','Red_Mean','Green_Mean','NIR_Mean','NDVI_Mean','Blue_std','Red_std','Green_std','NIR_std','NDVI_std']
cleanedDataDel[scaler_cols] = StandardScaler().fit_transform(cleanedDataDel[scaler_cols])
# cleanedDataDel[scaler_cols] =cleanedDataDel.groupby('Date')[scaler_cols].transform(lambda x: minmax_scale(x.astype(float)))
cleanedDataDel=cleanedDataDel.sort_values(by=['Date'])
cleanedDataDel['year']=0

trainData = cleanedDataDel[cleanedDataDel['Quadrat'].isin(Q_train)].rename(columns={"Quadrat": "id"})
testData = cleanedDataDel[cleanedDataDel['Quadrat'] .isin(Q_test)].rename(columns={"Quadrat": "id"})

seq_cols=['col_0',     'col_1',     'col_2','col_3',     'col_4',     'col_5']
seq_cols=seq_cols+['S2','S3','S4','year','Blue_Mean','Red_Mean','Green_Mean','NIR_Mean','NDVI_Mean','Blue_std','Red_std','Green_std','NIR_std','NDVI_std']
# generate X_train
X_train1=np.concatenate(list(list(gen_sequence(trainData[trainData['id']==id], seq_length, seq_cols)) for id in trainData['id'].unique()))
# print(X_train1.shape)
# generate y_train
y_train1=np.concatenate(list(list(gen_label(trainData[trainData['id']==id], seq_length, seq_cols,'Inc')) for id in trainData['id'].unique()))
y_train1=y_train1.astype('int')
print(y_train1.shape)
# generate X_test
X_test1=np.concatenate(list(list(gen_sequence(testData[testData['id']==id], seq_length, seq_cols)) for id in testData['id'].unique()))
# print(X_test1.shape)
# generate y_test
y_test1=np.concatenate(list(list(gen_label(testData[testData['id']==id], seq_length, seq_cols,'Inc')) for id in testData['id'].unique()))
y_test1=y_test1.astype('int')
print(y_test1.shape)
nb_features =X_train1.shape[2]
timestamp=seq_length


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
df = cleanedData.groupby('Quadrat').cumcount()
uniqueQuadrat2=cleanedData['Quadrat'].unique()
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

def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    if seq_length<6:
        lstm_array.append(data_array[5-seq_length:5, :])
        lstm_array.append(data_array[6-seq_length:6, :])
    else:
        lstm_array.append(data_array[0:6, :])
    return np.array(lstm_array)
def gen_label(id_df, seq_length, seq_cols,label):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label=[]
    if seq_length<6:
        y_label.append(id_df[label].values[4])
        y_label.append(id_df[label].values[5])
    else:
        y_label.append(id_df[label].values[5])
    return np.array(y_label)
dropCol = ['Blue', 'Green', 'Red', 'NIR', 'NDVI']
cleanedDataDel=cleanedData.drop(columns=dropCol).drop_duplicates().reset_index(drop=True)
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 7, 5).date(),'Date']=0
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 7, 9).date(),'Date']=1
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 7, 20).date(),'Date']=2
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 7, 31).date(),'Date']=3
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 8, 18).date(),'Date']=4
cleanedDataDel.loc[cleanedDataDel.Date==pd.datetime(2017, 8, 23).date(),'Date']=5

scaler_cols=['Blue_Mean','Red_Mean','Green_Mean','NIR_Mean','NDVI_Mean','Blue_std','Red_std','Green_std','NIR_std','NDVI_std']
cleanedDataDel[scaler_cols] = StandardScaler().fit_transform(cleanedDataDel[scaler_cols])
# cleanedDataDel[scaler_cols] =cleanedDataDel.groupby('Date')[scaler_cols].transform(lambda x: minmax_scale(x.astype(float)))
cleanedDataDel=cleanedDataDel.sort_values(by=['Date','Quadrat'])
cleanedDataDel['year']=1
one_hot_2 = pd.get_dummies(cleanedDataDel['Date'],prefix='col')
# Drop column B as it is now encoded
cleanedDataDel = cleanedDataDel.join(one_hot_2)
# cleanedDataDel=cleanedDataDel[cleanedDataDel.Date>=seq_length-1]
trainData = cleanedDataDel[cleanedDataDel['Quadrat'] .isin(Q_train)].rename(columns={"Quadrat": "id"})
testData = cleanedDataDel[cleanedDataDel['Quadrat'].isin(Q_test)].rename(columns={"Quadrat": "id"})



# generate X_train
X_train2=np.concatenate(list(list(gen_sequence(trainData[trainData['id']==id], seq_length, seq_cols)) for id in trainData['id'].unique()))
print(X_train2.shape)
# generate y_train
y_train2=np.concatenate(list(list(gen_label(trainData[trainData['id']==id], seq_length, seq_cols,'Inc')) for id in trainData['id'].unique()))
y_train2=y_train2.astype('int')
print(y_train2.shape)
# generate X_test
X_test2=np.concatenate(list(list(gen_sequence(testData[testData['id']==id], seq_length, seq_cols)) for id in testData['id'].unique()))
print(X_test2.shape)
# generate y_test
y_test2=np.concatenate(list(list(gen_label(testData[testData['id']==id], seq_length, seq_cols,'Inc')) for id in testData['id'].unique()))
y_test2=y_test2.astype('int')
print(y_test2.shape)
nb_features =X_train2.shape[2]
timestamp=seq_length

X_train=np.concatenate((X_train1,X_train2))
y_train=np.concatenate((y_train1,y_train2))
X_test=np.concatenate((X_test1,X_test2))
y_test=np.concatenate((y_test1,y_test2))
with open('dataGRU'+str(seq_length)+'.pickle', 'wb') as f:
    pickle.dump([X_train, X_test, y_train,y_test], f)
