# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
df= pd.read_csv('wine.csv')


'''
for c in df.columns:
    if df[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))

'''

train, test = train_test_split(df, test_size=0.2)




y_train = train["quality"]
y_test = test["quality"]

train = train.drop('quality', axis=1)
test = test.drop('quality',axis=1)

import xgboost as xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test)


# Uncomment to tune XGB `num_boost_rounds`
# Uncomment to tune XGB `num_boost_rounds`

cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=10000, early_stopping_rounds=20,
   verbose_eval=True, show_stdv=False)

num_boost_rounds = len(cv_result)
print(num_boost_rounds)
# num_boost_round = 489

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)



output = pd.DataFrame({'x': y_test, 'y': y_pred})

print mean_squared_error(y_test,y_pred) 

#print output


#output.to_csv('submit1.csv', index=False)
