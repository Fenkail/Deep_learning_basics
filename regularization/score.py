from sklearn.metrics import mean_squared_error


# 自定义评价函数
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label, preds)
    return 'myFeval', score

def out(test_sub, predictions):
    result=list(predictions)
    result=list(map(lambda x: x + 1, result))
    test_sub["happiness"]=result
    test_sub.to_csv("submit.csv", index=False)