import pandas as pd
from pandas.io.parsers import read_table

import numpy as np

import sklearn as sk

filename = "../dataset/simpleadult"

data = read_table(filename, sep=",")

#data = data.as_matrix()

#print data.as_matrix()
data = data.ix[:,:]

print data

data = data.T.to_dict().values
print data

#from sklearn.preprocessing import OneHotEncoder

#enc = OneHotEncoder()
#enc.fit(data)

#print enc.n_values_
#print enc.feature_indices_

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
data = dv.fit_transform(data)
print data.shape
print data[0]
print data.toarray()
