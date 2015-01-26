import os
import sys
import numpy as np
from time import time
#abspath = os.path.abspath(__file__)

CUR_WORK_DIR = os.getcwd()
sys.path.append(CUR_WORK_DIR)

import sklearn
from sklearn.tree import NBTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

from preprocessor import Preprocessor

def nbtree_test(
        X, y, X_test, y_test, meta,

        is_discretize    = False,
        diffprivacy_mech = "exp",
        budget           = 10., 

        criterion        = "gini", 
        max_depth        = 10,
        max_features     = 14,
        min_samples_leaf = 1,
        is_prune         = True,

        print_tree       = False,
        debug            = False,
        random_state     = 1000,
        output_file      = None):

    # redirect output to file
    if output_file is None:
        sys.stdout = sys.__stdout__
    else:
        sys.stdout = open(output_file, 'a')
    
    print "---------------------------------------------"
    if is_discretize:
        dchar = 'd'
    else:
        dchar = 'c'
    print "samples\tfeatures\tmech\tdisc\tbudget\tcriterion"
    print "%6dK\t%8d\t%4s\t%4s\t%6.2f\t%s"%(
            X.shape[0]//1000, X.shape[1], diffprivacy_mech, dchar, budget, criterion)
    print "---------------------------------------------"

    #print "mech\t",  diffprivacy_mech
    #print "budget\t\t",     budget
    #print "discretize\t",   is_discretize
    #print "max_depth\t",    max_depth
    #print "max_ftures\t",   max_features
    #print "criterion\t",    criterion
    #print "is prune\t",     is_prune
    #print "output\t\t",       output_file
    #print "print_tree\t",  print_tree
    #print "debug\t\t",     debug

    t1 = time()
    if diffprivacy_mech is not "org":
        nbtree = NBTreeClassifier(
                diffprivacy_mech= diffprivacy_mech, 
                budget          = budget, 

                criterion       = criterion, 
                max_depth       = max_depth, 
                max_features    = max_features,
                min_samples_leaf= min_samples_leaf,
                is_prune        = is_prune,
                random_state    = random_state,
                print_tree      = print_tree, 
                debug           = debug)

        nbtree.set_meta(meta)
        nbtree.fit(X,y)
        clf = nbtree
    else:
        tree = DecisionTreeClassifier(max_depth=max_depth, 
                                      random_state = random_state)
        tree = tree.fit(X, y )
        clf  = tree
    t2 = time()
    #print "fitting costs %.2fs"%(t2-t1)
    return clf

def evaluate( clf, X_test, y_test):

    y_true = y_test
    y_prob = clf.predict_proba(X_test)[:,-1]
    y_pred = clf.predict( X_test )

    score  = metrics.accuracy_score(    y_test, y_pred)
    auc    = metrics.roc_auc_score(     y_true, y_prob)
    matrix = metrics.confusion_matrix(  y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred, 
                        target_names=["label 0", "label 1"])

    print "Score:\t%.5f"%score
    print "AUC:  \t%.5f"%auc

    #print "Matrix:" 
    #print matrix

    return score, auc


def preprocess( is_load_from_raw=False, is_discretize=False, dataset="liantong", dmethod="cluster"):

    feature_in     = os.path.join(CUR_WORK_DIR, "dataset/feature.in")

    if dataset == "liantong":
        train_data_in  = os.path.join(CUR_WORK_DIR, "dataset/0506/05_cln.npy")
        test_data_in   = os.path.join(CUR_WORK_DIR, "dataset/0506/06_cln.npy")
    else:
        train_data_in  = os.path.join(CUR_WORK_DIR, "dataset/adult.data")
        test_data_in   = os.path.join(CUR_WORK_DIR, "dataset/adult.test")

    if is_discretize: 
        feature_out    = os.path.join(CUR_WORK_DIR, "dataset/feature_d.out")
        train_data_out = os.path.join(CUR_WORK_DIR, "dataset/data_d.out.npy")
        test_data_out  = os.path.join(CUR_WORK_DIR, "dataset/test_d.out.npy")
    else:
        feature_out    = os.path.join(CUR_WORK_DIR, "dataset/feature_c.out") 
        train_data_out = os.path.join(CUR_WORK_DIR, "dataset/data_c.out.npy")
        test_data_out  = os.path.join(CUR_WORK_DIR, "dataset/test_c.out.npy")

    preprocessor = Preprocessor()
    if is_load_from_raw:
        preprocessor.load( feature_in, train_data_in, test_data_in, sep=" ", 
                            is_discretize= is_discretize, 
                            nbins=10, 
                            dmethod=dmethod)

        preprocessor.export( feature_out, train_data_out, test_data_out)

    else:
        preprocessor.load_existed(feature_out, train_data_out, test_data_out)

    return preprocessor

if __name__ == "__main__":

    #m = "exp"
    #d = True
    dmethod = "bin"
    for d in [True, False]:
    #for c in ["gini", "entropy"]:
        c = "gini"
        for m in ["lap", "exp"]:

            if m is "lap" and d is False:
                continue
            
            p = preprocess( True, d,  "adult", dmethod = dmethod)

            X,      y      = p.get_train_X_y()
            X_test, y_test = p.get_test_X_y()
            features       = p.features

            is_prune = True
            #if m is "lap":
            #    is_prune = False

            clf = nbtree_test(X, y, X_test, y_test, features, 
                    max_depth = 5,
                    budget = 10.0,
                    is_discretize = d,
                    diffprivacy_mech = m,
                    criterion = c,
                    is_prune = is_prune,

                    print_tree = False
                    )

            evaluate(clf, X_test, y_test)
