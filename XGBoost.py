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
seq_length=4

with open('dataXGBoost'+str(seq_length)+'.pickle', 'rb') as f:
    X_train, X_test, y_train, y_test=pickle.load(f)
result=np.zeros((20,16))
result_2=np.zeros((40,y_test.shape[0]))
for i in range(1):

    class_weight = 1+0.2*i
    params = {'eta': 0.3,
              'tree_method': "hist",
              'grow_policy': "lossguide",
              'max_leaves': 40,
              'max_depth': 10,
              'subsample': 0.3,
              'colsample_bytree': 0.6,
              'colsample_bylevel': 0.6,
              'min_child_weight': 1,
              'alpha': 4,
              'objective': 'binary:logistic',
              'scale_pos_weight': class_weight,
              'eval_metric': 'rmse',
              'nthread': 1,
              'random_state': 99,
              'silent': True}
    model = xgb.train(params, dtrain, 300, watchlist, early_stopping_rounds=10, verbose_eval=5)

    # training metrics
    # scores = model.evaluate(X_train, y_train, verbose=0, batch_size=100)
    # print('train loss: {}'.format(scores))
    pred_train = model.predict(xgb.DMatrix(X_train), ntree_limit=model.best_ntree_limit)
    pred_train[pred_train<0.5]=0
    pred_train[pred_train>=0.5]=1
    acc_train=accuracy_score(y_train,pred_train)
    print('gru train acc: ',acc_train)
    mat_train=confusion_matrix(y_train,pred_train)
    print('Confusion Matrix: \n',confusion_matrix(y_train,pred_train))

    result[i][0] = mat_train[0][0]
    result[i][1] = mat_train[0][1]
    result[i][2] = mat_train[1][0]
    result[i][3] = mat_train[1][1]

    pred_test = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)
    result_2[2*i][:]=pred_test.T
    result_2[2 * i+1][:] = y_test.T
    pred_test [pred_test<0.5]=0
    pred_test [pred_test>=0.5]=1
    acc_test = accuracy_score(y_test, pred_test)
    print('test acc: ',acc_test)
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
