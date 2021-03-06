## 实验目标

- 基于**个体变量**、**家庭变量**、**社会态度**等指标，来预测其对**幸福感**的强度。
- 以XGBOOST作为实验模型，验证L1、L2、类似Dropout的方式进行正则，期待增强模型的泛化能力在验证集上取得更好的效果。

## 实验过程

初始参数:      验证集得分为0.45434592

```
xgb_params = {"booster": 'gbtree',
              'eta': 0.005,
              'max_depth': 5,
              'subsample': 0.7,          # 随机采样训练样本
              'colsample_bytree': 0.8,
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'silent': True,
              }
```

增加'alpha'作为L1正则参数， 'lambda'作为L2正则的参数，'subsample'为类Dropout的作用。

### 实验一：验证L1正则的作用

| 方法             | 得分       |      |
| ---------------- | ---------- | ---- |
| 初始参数（base） | 0.45434592 |      |
| 'alpha'=1        | 0.45460193 |      |
| 'alpha'=0.7      | 0.45447847 |      |
| 'alpha'=0.6      | 0.45467713 |      |
| 'alpha'=0.55     | 0.45430484 | 最优 |
| 'alpha'=0.5      | 0.45431723 |      |
| 'alpha'=0.3      | 0.45450940 |      |

### 实验二：验证L2正则的作用

| 方法                       | 得分       |      |
| -------------------------- | ---------- | ---- |
| 初始参数（base）           | 0.45434592 |      |
| 'lambda'=3                 | 0.45484879 |      |
| 'lambda'=2                 | 0.45488106 |      |
| 'lambda'=1                 | 0.45434592 |      |
| 'lambda'=0.5               | 0.45469010 |      |
| 'alpha'=0.55，'lambda'=0.5 | 0.45467713 |      |
| 'alpha'=0.55，'lambda'=3   | 0.45431729 |      |

### 实验三：验证类Dropout的作用

修改随机采样训练样本的权重'subsample'，每一次训练随机采样一定比例的样本，使得模型训练每次输入数据不同，增强模型的泛化能力。dropout是在每次训练的时候随机丢弃某些神经元，和数据的随机采样，数据增强有着相似的作用。

| 方法                | 得分       |      |
| ------------------- | ---------- | ---- |
| 初始参数（base）0.7 | 0.45434592 |      |
| 'subsample': 0.6    | 0.45448684 |      |
| 'subsample': 0.8    | 0.45625735 |      |

## 结论

基于base增加L1正则，当选择到合适的值后会得到更好的模型效果。无论是基于base增加L2正则还是基于'alpha'=0.55增加L2后，模型的效果皆为下降的状态。

- 特征的维度为145，训练样本中整数参数较多且较多的值相等，减去同一偏差后，矩阵较为稀疏。-->L1的正则对稀疏特征有效。
- L2的正则在本次实验中未取得较好的效果，通常可能更适合较为稠密的特征

训练数据的随机采样*类似于*数据增强和Dropout的效果

- 本实验中训练数据随机采样的比例控制到0.7可以有效的提升泛化效果