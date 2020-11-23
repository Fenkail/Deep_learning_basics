import numpy as np
import time
import xgboost as xgb
from sklearn.model_selection import KFold
from regularization.score import myFeval
from sklearn.metrics import mean_squared_error
from regularization.data_process import data_process

s = time.time()
X_train, y_train, X_test, train_len, test_len = data_process()
print('加载数据及预处理消耗时间：', time.time()-s)

xgb_params = {"booster": 'gbtree',
              'eta': 0.005,
              'max_depth': 5,
              'subsample': 0.7, # 随机采样训练样本
              'colsample_bytree': 0.8,
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'silent': True,
              'lambda': 1
              }
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(train_len)
predictions_xgb = np.zeros(test_len)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params, feval=myFeval)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train.tolist())))



'''
 --------------------------------------------
1. 初始参数  reg:linear     0.45434592
2. 增加L2正则 'lambda':2    0.45488106
3. 2+增加L1正则 'alpha': 1  0.45456481
4. 增加L1正则 'alpha': 1    0.45460193
5. 3+subsample改为0.6      0.45449627
6. 只改subsample 0.6       0.45448684
7. 只改subsample 0.8       0.45625735
8. 1+增加L1正则0.5          0.45431723         
9. 1+增加L1正则0.3          0.45450940
10.1+增加L1正则0.7          0.45447847
11.1+增加L1正则0.6          0.45467713
12.1+增加L1正则0.55         0.45430484            ○
13.12+L2正则0.5            0.45467713
14.12+L2正则3              0.45431729
15.1+增加L2正则3            0.45484879
16.1+增加L2正则1            0.45434592
17.1+增加L2正则0.5          0.45469010




'''