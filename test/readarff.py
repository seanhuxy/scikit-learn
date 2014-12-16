import os
from scipy.io.arff import loadarff
import numpy as np
from dict_vect import Dict_Vectorizer

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import NBTreeClassifier
from sklearn.cross_validation import cross_val_score

filename = os.getenv("HOME")+"/diffprivacy/dataset/adult-census.arff"

data, meta = loadarff(filename)

data_dict = [dict(zip(data.dtype.names, record)) for record in data]

vectorizer = Dict_Vectorizer()
X, y, meta = vectorizer.fit_transform(data_dict, None)

nbtree = NBTreeClassifier()
nbtree = nbtree.fit(X,y,meta)

output =  cross_val_score(nbtree, X, y, cv=10)


print "test finished"



