import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

houses = pd.read_csv("Adult_train.tab", sep='\t', skiprows=[1, 2])

y = (houses['y'] == '>50K').astype('int')

x = houses.drop(['y'], axis=1)
x = x.drop(['education'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
