import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

houses = pd.read_csv("Adult_train.tab", sep='\t', skiprows=[1, 2])

y = (houses['y'] == '>50K').astype('int')

x = houses.drop(['y'], axis=1)
x = x.drop(['education'], axis=1)

one_hot_vars = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

x = pd.get_dummies(x, columns=one_hot_vars)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x))


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(len(x_train))
print(len(x_train.columns))

# # svm
# svc_params = {
#     'C': [1, 10],
# }
#
# clf = GridSearchCV(SVC(max_iter=10000), svc_params, cv=5, verbose=3, n_jobs=-1)

# decision trees
# tree_params = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [5, 6, 8, 10, 20],
#     'splitter': ['best', 'random']
# }
# clf = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=5)

random_forest_params = {
    'n_estimators': [100],
    'max_depth': [21],
}
clf = GridSearchCV(RandomForestClassifier(), random_forest_params, cv=5)


clf.fit(x_train, y_train)
print(clf.best_params_)

y_train_pred = clf.predict(x_train)
print(classification_report(y_train, y_train_pred))

y_test_pred = clf.predict(x_test)
print(classification_report(y_test, y_test_pred))

