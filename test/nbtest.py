import os
from scipy.io.arff import loadarff
import numpy as np
from dict_vect import Dict_Vectorizer

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import NBTreeClassifier
from sklearn.cross_validation import cross_val_score

def test(dataset="adult.arff", 
        discretize = True, 
        max_depth = 5, 
        diffprivacy_mech = "no", 
        budget=-1.0, 
        criterion="entropy", 
        min_samples_leaf=0, 
        print_tree = False,
        is_prune = True):
    
    filename = os.getenv("HOME")+"/diffprivacy/dataset/"+dataset

    print "Building Tree for ", dataset
    print "discretize\t\t", discretize
    print "max_depth\t\t", max_depth
    print "min_samples_leaf\t", min_samples_leaf
    print "diffprivacy\t\t", diffprivacy_mech
    print "budget\t\t\t", budget
    print "criterion\t\t", criterion
    print "print_tree\t\t", print_tree
    print "is prune\t\t", is_prune

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
                is_prune = is_prune)
    # nbtree = nbtree.fit(X,y,meta)
    output =  cross_val_score(nbtree, X, y, cv=10, fit_params={'meta':meta, 'debug':False})

    print output
    print "Average Accuracy:", np.average(output)
    
    return np.average(output)


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


def test_nodp_gini():
    
    diffprivacy_mech = "no"
    criterion = "gini"

    accuracy = []
    print "Test Case: No diffprivacy, Criterion: gini"
    for i in range(1,11):
        ret = test(max_depth = i, diffprivacy_mech = diffprivacy_mech, criterion = criterion)
        accuracy.append(ret)

    for i in range(1,11):
        print "max_depth[{0}]   {1}".format(i, accuracy[i-1])

def debug_zero_nodp_gini():
    
    diffprivacy_mech = "no"
    criterion = "gini"

    ret = test(max_depth = 7, diffprivacy_mech = diffprivacy_mech, criterion = criterion)

    print "max_depth[{0}]   {1}".format(10, ret)


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
    criterion = "gini"
    max_depth = 5
    budgets = [0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0]

    accuracys = []
    print "Test Case: Exponential mech, Criterion: gini"
    for i in budgets:
        ret = test(max_depth = max_depth, 
                    diffprivacy_mech = diffprivacy_mech, 
                    budget = i,
                    criterion = criterion)
        accuracys.append(ret)

    for i in range(len(budgets)):
        print "budget[{0}]   {1}".format( budgets[i], accuracys[i])


def test_one_exp_gini():
    diffprivacy_mech = "exp"
    criterion = "gini"
    max_depth = 10
    budget =  5.0 

    accuracy = 0 
    print "Test Case: Exponential mech, Criterion: gini"
    ret = test(max_depth = max_depth, 
            diffprivacy_mech = diffprivacy_mech, 
            budget = budget,
            criterion = criterion)
    accuracy = ret

    print "budget{0}   {1}".format( budget, accuracy)




if __name__ == "__main__":
    # test_nodp_entropy()
    #test_nodp_gini()
#    test_lap_entropy()
#    test_exp_gini()
    #test_one_exp_gini()
    debug_zero_nodp_gini()
    #test()

