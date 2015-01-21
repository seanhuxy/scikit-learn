import os
import sys
import numpy as np
from time import time
#abspath = os.path.abspath(__file__)

cwd = os.getcwd()
sys.path.append(cwd)

import sklearn
from sklearn.tree import NBTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

from preprocessor import Preprocessor


def filter_feature( data, n_features ):

    #data = np.take( data, feature_importances[:n_features]))
    data = data[ : , feature_importances[ :n_features] ]
    return data

def filter_sample( data, n_samples ):
	
    data = data[ : n_samples]
    return data
    # or bagging


def main( data,  is_discretized )

    # get from a standard classifier algor
    tree = DecisionTreeClassifier(max_depth=max_depth)

    print "fitting..."
    t1 = time()
    tree.fit( X, y)
    clf = tree
    t2 = time()
    print "Time for fitting %.2fs"%(t2-t1)
    
    feature_importances = clf.feature_importances_
    feature_importances = np.argsort(feature_importances)
    print feature_importances
    for i, f in enumerate(features):
        print "[%2d]\t%25s\t%.3f"%(i, meta.features[f].name, feature_importances[f])
    print "\n"

    # criterion
    criterions = ["entropy", "gini"]

    # budgets
    budgets = [ 0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]

    # features: 70
    n_features = [  70, 60, 50, 40, 30, 20,  
                    17, 14, 11,  9,  7,  5 ]

    n_samples  = [  2000000, # 2.0M
                    1500000, # 1.5M
                    1000000, # 1.0M
                     500000, # 0.5M
                     100000, # 0.1M
                      50000, # 50K
                      10000 ]# 10K

    # discretized
    data = ... # discretized

    mech = "no"
    for f in n_features:
        for s in n_samples:
            data = top(s, f) 
            
            nbtree_test( data, 
                         diffprivacy_mech = mech,
                         budget           = -1.0,

                         max_depth        = max_depth,
                         is_prune         = is_prune,
                         output_file      = output_file)

    mechs = ["lap", "exp"]
    for mech in mechs:
        for f in n_features:
            for s in n_samples:
                for b in budgets:
                    data = top(s, f) 
                    
                    nbtree_test( data, 
                                 diffprivacy_mech = mech,
                                 budget           = b,

                                 max_depth        = max_depth,
                                 is_prune         = is_prune,
                                 output_file      = output_file)

    # not discretized
    data = ... # not discretized

    mech = "no"
    for f in n_features:
        for s in n_samples:
            data = top(s, f) 
            
            nbtree_test( data, 
                         diffprivacy_mech = mech,
                         budget           = -1.0,

                         max_depth        = max_depth,
                         is_prune         = is_prune,
                         output_file      = output_file)

    mech = "exp"
    for f in n_features:
        for s in n_samples:
            for b in budgets:
                data = top(s, f) 
                
                nbtree_test( data, 
                             diffprivacy_mech = mech,
                             budget           = b,

                             max_depth        = max_depth,
                             is_prune         = is_prune,
                             output_file      = output_file)









