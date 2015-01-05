import os
import sys
print __file__

sys.path.append(os.getcwd())
print sys.path

from preprocessor import Preprocessor

import sklearn
from sklearn.tree import NBTreeClassifier
from sklearn.cross_validation import cross_val_score

import numpy as np

feature_in = os.getenv("HOME")+"/diffprivacy/dataset/adult_nomissing.arff"
#feature_in = "test/feature.in"
data_in = "test/data.in"

feature_out = "test/feature.out"
data_out = "test/data.out"

preprocessor = Preprocessor()

preprocessor.load( feature_in, data_in)
preprocessor.discretize(nbins=10)
preprocessor.export( feature_out, data_out)

X = preprocessor.get_X()
y = preprocessor.get_y()

print X
print y

def test(X, y, meta,
        discretize = False,
        max_depth = 10,
        diffprivacy_mech = "exp",
        budget =100000.0, 
        criterion="gini", 
        min_samples_leaf=0, 
        print_tree = True,
        is_prune = True,
        debug = False,
        seed = 2):

    print "# ==================================="
    print "diffprivacy\t", diffprivacy_mech
    print "budget\t\t", budget
    print "discretize\t", discretize
    print "max_depth\t", max_depth
    print "criterion\t", criterion
    print "print_tree\t", print_tree
    print "is prune\t", is_prune
    print "debug\t\t", debug
    print "seed\t\t", seed

    nbtree = NBTreeClassifier(
                max_depth=max_depth, 
                diffprivacy_mech=diffprivacy_mech, 
                criterion=criterion, 
                budget=budget, 
                print_tree=print_tree, 
                min_samples_leaf=min_samples_leaf,
                is_prune = is_prune,
                seed = seed)

    #nbtree = nbtree.fit(X,y,meta, debug = debug)
    output =  cross_val_score(nbtree, X, y, cv=5, fit_params={'meta':meta, 'debug':debug})

    print output
    print "Average Accuracy:", np.average(output)
    print "# =========================================" 
    print "\n"


test(X, y, preprocessor)
