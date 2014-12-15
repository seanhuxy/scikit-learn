import os
from scipy.io.arff import loadarff
import numpy as np
from dict_vect import Dict_Vectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import NBTreeClassifier

filename = os.getenv("HOME")+"/diffprivacy/dataset/adult-census.arff"

data, meta = loadarff(filename)

data_dict = [dict(zip(data.dtype.names, record)) for record in data]

vectorizer = Dict_Vectorizer()
X, y, meta = vectorizer.fit_transform(data_dict, None)

nbtree = NBTreeClassifier()
nbtree = nbtree.fit(X,y,meta)





