import pandas as pd
from sklearn.preprocessing import OneHotEncoder

houses = pd.read_csv("Adult_train.tab", sep='\t', skiprows=[1, 2])

y_train = (houses['y'] == '>50K').astype('int')

x_train = houses.drop(['y'], axis=1)
x_train = x_train.drop(['education'], axis=1)


def get_one_hot_repr(data, colname):
    ohe = OneHotEncoder()
    result = ohe.fit_transform(data[colname].values.reshape(-1, 1)).toarray()
    result = pd.DataFrame(result, columns=[colname + str(ohe.categories_[0][i]) for i in range(len(ohe.categories_[0]))])
    return result


def substitute_with_one_hot(data, colnames):
    for colname in colnames:
        one_hot_column = get_one_hot_repr(data, colname)
        one_hot_column.fillna(0)
        data = pd.concat([data, one_hot_column], axis=1)
        del data[colname]
    return data

print(x_train['education'].head())