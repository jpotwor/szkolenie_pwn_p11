import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

houses = pd.read_csv("Adult_train.tab", sep='\t', skiprows=[1, 2])

y = (houses['y'] == '>50K').astype('int')

x = houses.drop(['y'], axis=1)
x = x.drop(['education'], axis=1)

one_hot_vars = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

x = pd.get_dummies(x, columns=one_hot_vars)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scale_cols = ['age', 'fnlwgt', 'education-num', 'capital-loss', 'hours-per-week']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(len(x_train))
print(len(x_train.columns))

svc_params = {
    'kernel': ['rbf'],
    'gamma': [1e-3, 1e-4],
    'C': [1, 10, 100, 1000]
}
clf = GridSearchCV(SVC(), svc_params, cv=5, verbose=3)

clf.fit(x_train, y_train)
print(clf.best_params_)

y_train_pred = clf.predict(x_train)
print(classification_report(y_train, y_train_pred))
