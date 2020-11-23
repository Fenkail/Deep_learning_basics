from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from regularization.data_process import  data_process

X_train, y_train, X_test, train_len, test_len = data_process()

# 将lgb和xgb的结果进行stacking
#TODO oof_lgb, oof_xgb，predictions_lgb, predictions_xgb 从上述两个模型导入，或者先存储，在读取
oof_lgb, oof_xgb = None
predictions_lgb, predictions_xgb = None
train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2018)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]

    clf_3 = linear_model.BayesianRidge()
    # clf_3 =linear_model.Ridge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10

print("CV score: {:<8.8f}".format(mean_squared_error(oof_stack, y_train.tolist())))
