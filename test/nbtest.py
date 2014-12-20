import os
from scipy.io.arff import loadarff
import numpy as np
from dict_vect import Dict_Vectorizer

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import NBTreeClassifier
from sklearn.cross_validation import cross_val_score

filename = os.getenv("HOME")+"/diffprivacy/dataset/adult.arff"

data, meta = loadarff(filename)

data_dict = [dict(zip(data.dtype.names, record)) for record in data]

vectorizer = Dict_Vectorizer()
X, y, meta = vectorizer.fit_transform(data_dict, None)


if y.shape[1] == 1:
    y = np.squeeze(y)
print y.shape

print "X.dtype:", X.dtype
print "Y.dtype:", y.dtype

nbtree = NBTreeClassifier(
            max_depth=3, 
            diffprivacy_mech="exp", 
            criterion="gini", 
            budget=100.0, 
            print_tree=True, 
            min_samples_leaf=0)

# nbtree = nbtree.fit(X,y,meta)

output =  cross_val_score(nbtree, X, y, cv=10, fit_params={'meta':meta, 'debug':False})

print output
print "test finished"



