import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_regression

s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(s)
new_index = ['x', 'y', 'z', 'w']
s_new = s.reindex(new_index)
print(s_new)
# s = pd.concat([dummy_features, new_s], ignore_index=True, axis=1)
# print(s)

