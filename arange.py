import pandas as pd
import numpy as np

students = pd.read_csv('students.csv')
students = students.sample(frac=1)
students['ind'] = np.arange(len(students['name']))
students['group'] = students['ind'] % 3

print(students.sort_values(by=['group']))
print(students.groupby(['group'])['name'].min())


