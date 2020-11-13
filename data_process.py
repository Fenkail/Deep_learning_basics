import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, RepeatedKFold
from scipy import sparse

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
from datetime import datetime


# 把一天的时间分段
def hour_cut(x):
    if 0 <= x < 6:
        return 0
    elif 6 <= x < 8:
        return 1
    elif 8 <= x < 12:
        return 2
    elif 12 <= x < 14:
        return 3
    elif 14 <= x < 18:
        return 4
    elif 18 <= x < 21:
        return 5
    elif 21 <= x < 24:
        return 6

def birth_split(x):
    if 1920 <= x <= 1930:
        return 0
    elif 1930 < x <= 1940:
        return 1
    elif 1940 < x <= 1950:
        return 2
    elif 1950 < x <= 1960:
        return 3
    elif 1960 < x <= 1970:
        return 4
    elif 1970 < x <= 1980:
        return 5
    elif 1980 < x <= 1990:
        return 6
    elif 1990 < x <= 2000:
        return 7

def income_cut(x):
    if x < 0:
        return 0
    elif 0 <= x < 1200:
        return 1
    elif 1200 < x <= 10000:
        return 2
    elif 10000 < x < 24000:
        return 3
    elif 24000 < x < 40000:
        return 4
    elif 40000 <= x:
        return 5

def data_process():
    # 导入数据
    train_abbr = pd.read_csv("dataset/happiness/happiness_train_abbr.csv", encoding='ISO-8859-1')
    train = pd.read_csv("dataset/happiness/happiness_train_complete.csv", encoding='ISO-8859-1')
    test_abbr = pd.read_csv("dataset/happiness/happiness_test_abbr.csv", encoding='ISO-8859-1')
    test = pd.read_csv("dataset/happiness/happiness_test_complete.csv", encoding='ISO-8859-1')
    test_sub = pd.read_csv("dataset/happiness/happiness_submit.csv", encoding='ISO-8859-1')

    # 查看label分布
    y_train_ = train["happiness"]
    # y_train_.value_counts()
    y_train_ = y_train_.map(lambda x: 3 if x == -8 else x)
    y_train_ = y_train_.map(lambda x: x - 1)
    data = pd.concat([train, test], axis=0, ignore_index=True)
    # 数据预处理
    data['survey_time'] = pd.to_datetime(data['survey_time'], format='%Y-%m-%d %H:%M:%S')
    data["weekday"] = data["survey_time"].dt.weekday
    data["year"] = data["survey_time"].dt.year
    data["quarter"] = data["survey_time"].dt.quarter
    data["hour"] = data["survey_time"].dt.hour
    data["month"] = data["survey_time"].dt.month
    data["hour_cut"] = data["hour"].map(hour_cut)
    data["survey_age"] = data["year"] - data["birth"]
    data["happiness"] = data["happiness"].map(lambda x: x - 1)

    #去掉三个缺失值很多的
    data=data.drop(["edu_other"], axis=1)
    data=data.drop(["happiness"], axis=1)
    data=data.drop(["survey_time"], axis=1)

    data["join_party"] = data["join_party"].map(lambda x:0 if pd.isnull(x)  else 1)
    data["birth_s"] = data["birth"].map(birth_split)


    data["income_cut"] = data["income"].map(income_cut)
    #填充数据
    data["edu_status"]=data["edu_status"].fillna(5)
    data["edu_yr"]=data["edu_yr"].fillna(-2)
    data["property_other"]=data["property_other"].map(lambda x:0 if pd.isnull(x)  else 1)
    data["hukou_loc"]=data["hukou_loc"].fillna(1)
    data["social_neighbor"]=data["social_neighbor"].fillna(8)
    data["social_friend"]=data["social_friend"].fillna(8)
    data["work_status"]=data["work_status"].fillna(0)
    data["work_yr"]=data["work_yr"].fillna(0)
    data["work_type"]=data["work_type"].fillna(0)
    data["work_manage"]=data["work_manage"].fillna(0)
    data["family_income"]=data["family_income"].fillna(-2)
    data["invest_other"]=data["invest_other"].map(lambda x:0 if pd.isnull(x)  else 1)
    data["minor_child"]=data["minor_child"].fillna(0)
    data["marital_1st"]=data["marital_1st"].fillna(0)
    data["s_birth"]=data["s_birth"].fillna(0)
    data["marital_now"]=data["marital_now"].fillna(0)
    data["s_edu"]=data["s_edu"].fillna(0)
    data["s_political"]=data["s_political"].fillna(0)
    data["s_hukou"]=data["s_hukou"].fillna(0)
    data["s_income"]=data["s_income"].fillna(0)
    data["s_work_exper"]=data["s_work_exper"].fillna(0)
    data["s_work_status"]=data["s_work_status"].fillna(0)
    data["s_work_type"]=data["s_work_type"].fillna(0)
    data = data.drop(["id"], axis=1)

    X_train_ = data[:train.shape[0]]
    X_test_  = data[train.shape[0]:]

    target_column = 'happiness'
    feature_columns=list(X_test_.columns)

    X_train = np.array(X_train_)
    y_train = np.array(y_train_)
    X_test  = np.array(X_test_)
    return X_train, y_train, X_test, len(train), len(test)