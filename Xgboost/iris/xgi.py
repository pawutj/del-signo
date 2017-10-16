# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
df= pd.read_csv('iris.csv')

for c in df.columns:
    if df[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))

train, test = train_test_split(df, test_size=0.2)

print train.head()




y_train = train["species"]
y_test = test["species"]

train = train.drop('species', axis=1)
test = test.drop('species',axis=1)

'''
xgb = XGBClassifier()
xgb.fit(train, y_train)
y_pred = xgb.predict(test)

#print y_pred
#print y_test
'''


rfo = RandomForestClassifier()
ada = AdaBoostClassifier()
ext = ExtraTreesClassifier()
xgb = XGBClassifier()

vote_list = [('rfo', rfo), ('ada', ada), ('ext', ext), ('xgb', xgb)]
vote = VotingClassifier(estimators=vote_list, voting='soft')
vote.fit(train, y_train)
score = vote.score(train, y_train)
y_pred = vote.predict(test)

output = pd.DataFrame({'x':y_pred,'y':y_test})
print output

'''

output.to_csv('submit1.csv', index=False)
'''