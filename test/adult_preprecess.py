import pandas as pd
import numpy as np
from preprocessor import Preprocessor
import os

dataset = os.getcwd()+"/dataset"

adult_data = "adult.data"
adult_test = "adult.test"
adult_feature = "adult.feature"

feature_file = os.path.join( dataset, adult_feature)
data_file = os.path.join( dataset, adult_data)
test_file = os.path.join( dataset, adult_test)

p = Preprocessor()
features = p._load_features( feature_file )

names = [ f.name for f in features]
print names

d = pd.read_csv( data_file, names=names, sep=", ", header=None)
data = np.array(d)

d = pd.read_csv( test_file, names=names, sep=", ", header=None)
test = np.array(d)

def remove_missing( array ):

    print "before elimiate missing value, shape:", array.shape
    del_row = []
    for i, row in enumerate(array):
        for col in row:
            if col == '?':
                del_row.append(i)

    array = np.delete( array, del_row, axis=0)
    print "after elimiate missing value, shape:", array.shape
    return array

data = remove_missing(data)
test = remove_missing(test)

def remove_phone_number(data):

    data = np.delete( data, 0, axis=1)
    return data












