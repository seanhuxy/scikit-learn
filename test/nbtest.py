import os
from scipy.io.arff import loadarff
import numpy as np
from dict_vect import Dict_Vectorizer

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import NBTreeClassifier
from sklearn.cross_validation import cross_val_score

def test(dataset="adult_nomissing.arff", 
        discretize = True, 
        max_depth = 5, 
        diffprivacy_mech = "no", 
        budget=-1.0, 
        criterion="entropy", 
        min_samples_leaf=0, 
        print_tree = True,
        is_prune = True,
        debug = False,
        seed = 2):

    
    filename = os.getenv("HOME")+"/diffprivacy/dataset/"+dataset

    print "# ==================================="
    print "dataset\t", dataset
    print "diffprivacy\t", diffprivacy_mech
    print "budget\t\t", budget
    print "discretize\t", discretize
    print "max_depth\t", max_depth
    print "criterion\t", criterion
    print "print_tree\t", print_tree
    print "is prune\t", is_prune
    print "debug\t\t", debug
    print "seed\t\t", seed

    data, meta = loadarff(filename) 
    data_dict = [dict(zip(data.dtype.names, record)) for record in data] 
#    print meta
    vectorizer = Dict_Vectorizer()
    X, y, meta = vectorizer.fit_transform(data_dict, None, discretize=discretize)

    if y.shape[1] == 1:
        y = np.squeeze(y)

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

    #return np.average(output)


def test_nodp_entropy():

    diffprivacy_mech = "no"
    criterion = "entropy"

    accuracy = []
    print "Test Case: No diffprivacy, Criterion: entropy"
    for i in range(1,11):
        ret = test(max_depth = i, diffprivacy_mech = diffprivacy_mech, criterion = criterion)
        accuracy.append(ret)

    for i in range(1,11):
        print "max_depth[{0}]   {1}".format(i, accuracy[i-1])


def test_nominal_nodp_gini():
     
    depth = [1,3,5,7,9]
    accuracy = []
    print "Test Case: No diffprivacy, Criterion: gini"
    for i in range(len(depth)):
        ret = test(max_depth = depth[i], 
                    discretize = True,
                    diffprivacy_mech = "no",
                    budget = -1.0,
                    criterion = "gini",
                    print_tree = True,
                    is_prune = False)
        accuracy.append(ret)

    for i in range(len(depth)):
        print "max_depth[{0}]   {1}".format( depth[i] , accuracy[i])

def test_lap_entropy():
    
    diffprivacy_mech = "lap"
    criterion = "entropy"
    max_depth = 5
    budgets = [0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0]

    accuracys = []
    print "Test Case: Laplace mech, Criterion: entropy"
    for i in budgets:
        ret = test(max_depth = max_depth, 
                    diffprivacy_mech = diffprivacy_mech, 
                    budget = i,
                    criterion = criterion)
        accuracys.append(ret)

    for i in range(len(budgets)):
        print "budget[{0}]   {1}".format( budgets[i], accuracys[i])


def test_exp_gini():
    diffprivacy_mech = "exp"
    budgets = [ 1.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0]
    criterion = "gini"
    max_depth = 10
    min_samples_leaf = 2
    discretize = False
    debug = False

    accuracys = []
    print "Test Case: Exponential mech, Criterion: gini"
    for i in budgets:
        ret = test(max_depth = max_depth, 
                    diffprivacy_mech = diffprivacy_mech, 
                    budget = i,
                    criterion = criterion,
                    discretize = discretize)
        accuracys.append(ret)

    for i in range(len(budgets)):
        print "budget[{0}]   {1}".format( budgets[i], accuracys[i])


def test_one_exp_gini():
   
    budget = 10000000.0
    accuracy = 0 
    print "Test Case: Exponential mech, Criterion: gini"
    ret = test( 
            discretize = True,
            max_depth = 10,
            min_samples_leaf = 2,

            diffprivacy_mech = "exp",
            budget  =  budget,
            criterion = "gini",

            is_prune = True,

            seed = 9,
            print_tree = True,
            debug = False
            )

    accuracy = ret

    print "budget {0}   {1}".format( budget, accuracy)


def test_nodiscrete_nodp_gini():

    diffprivacy_mech = "no"
    budget = -1.0
    criterion = "gini"
    max_depth = 10
    
    discretize = False

    accuracy = 0 
    print "Test Case: Exponential mech, Criterion: gini"
    ret = test(max_depth = max_depth, 
            diffprivacy_mech = diffprivacy_mech, 
            budget = budget,
            criterion = criterion,
            discretize = False)
    accuracy = ret

    print "Accuracy  {0}".format(  accuracy)

if __name__ == "__main__":
    # test_nodp_entropy()
    #test_nominal_nodp_gini()
#    test_lap_entropy()
    # test_exp_gini()
    test_one_exp_gini()
    #debug_zero_nodp_gini()
    #test_nodiscrete_nodp_gini()
    #test()

