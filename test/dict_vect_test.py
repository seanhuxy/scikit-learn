import os
from scipy.io.arff import loadarff
import numpy as np
from dict_vect import Dict_Vectorizer


filename = os.getenv("HOME")+"/diffprivacy/dataset/mushroom.arff"

data, meta = loadarff(filename)

data_dict = [dict(zip(data.dtype.names, record)) for record in data]

v = Dict_Vectorizer()
X, y, meta = v.fit_transform(data_dict, None)

print "X", X.shape
print "y", y.shape

n_samples = X.shape[0]
n_features= X.shape[1]

for i in range( 10):
    for f in range(n_features):
        f_name  = v.feature_name(f)
        f_value = v.feature_value(np.int_(f),np.int_(X[i,f])) 
        print "%s:%s, "%(f_name,f_value),
    print "\n"


arr, count = np.unique(y,return_counts=True)
print arr,count
