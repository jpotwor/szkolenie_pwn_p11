import pandas as pd

houses = pd.read_csv("Adult_train.tab", sep='\t', skiprows=[1, 2])