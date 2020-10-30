import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import keras
import gc
import xgboost as xgb
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
with open('dataGRU'+str(seq_length)+'.pickle', 'rb') as f:
    X_train, X_test, y_train, y_test=pickle.load(f)
model = Sequential()
model.add(GRU(
         50,
         return_sequences=True, input_shape=(timestamp, nb_features)))
# model.add(Dropout(0.2))

model.add(GRU(
          units=25,
          return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='RMSprop',metrics=['accuracy'])

model.summary()
result=np.zeros((20,16))
result_2=np.zeros((40,y_test.shape[0]))
for i in range(1):
    class_weight = {0: 1.,
                1: 1+0.2*i}
    num_epoch = 4
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1)
    model.fit(X_train, y_train, epochs=num_epoch, batch_size=100,verbose=1,callbacks=[callback],class_weight=class_weight)
    pred_train=model.predict_classes(X_train)
    acc_train=accuracy_score(y_train,pred_train)
    print('gru train acc: ',acc_train)
    mat_train=confusion_matrix(y_train,pred_train)
    print('Confusion Matrix: \n',confusion_matrix(y_train,pred_train))

    result[i][0] = mat_train[0][0]
    result[i][1] = mat_train[0][1]
    result[i][2] = mat_train[1][0]
    result[i][3] = mat_train[1][1]
    pred_test=model.predict(X_test)
    result_2[2*i][:]=pred_test.T
    result_2[2 * i+1][:] = y_test.T
    pred_test=model.predict_classes(X_test)
    acc_test = accuracy_score(y_test, pred_test)
    print('gru test acc: ',acc_test)
    mat_test = confusion_matrix(y_test,pred_test)
    print('Confusion Matrix: \n',mat_test)

    result[i][4] = mat_test[0][0]
    result[i][5] = mat_test[0][1]
    result[i][6] = mat_test[1][0]
    result[i][7] = mat_test[1][1]
    result[i][8] = acc_train
    result[i][9] = acc_test
    result[i][10] = mat_train[1][1]/(mat_train[1][1]+mat_train[1][0])
    result[i][11] = mat_train[1][1] / (mat_train[1][1] + mat_train[0][1]+0.1)
    result[i][12] = mat_test[1][1] / (mat_test[1][1] + mat_test[1][0])
    result[i][13] = mat_test[1][1] / (mat_test[1][1] + mat_test[0][1]+0.1)
    result[i][14] = (mat_train[1][0]+mat_train[1][1])\
                     /(mat_train[1][0]+mat_train[1][1]
                       +mat_train[0][0]+mat_train[0][1])
    result[i][15] = (mat_test[1][0] + mat_test[1][1]) \
                    / (mat_test[1][0] + mat_test[1][1]
                       + mat_test[0][0] + mat_test[0][1])
result_csv=pd.DataFrame(data=result)

result_csv.columns=['train_mat00','train_mat01','train_mat10','train_mat11',
                    'test_mat00','test_mat01','test_mat10','test_mat11',
                    'train_acc','test_acc','train_recall','train_precision','test_recall','test_precision','train_rate','test_rate']
# result_csv.to_csv('result_GRU_2016and2017_'+str(seq_length)+'_400.csv',index=False)
