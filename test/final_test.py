import os
import sys
import numpy as np
from time import time
#abspath = os.path.abspath(__file__)

CUR_WORK_DIR= os.getcwd()
OUTPUT_DIR  = os.path.join(CUR_WORK_DIR, "log")  
sys.path.append(CUR_WORK_DIR)

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

def nbtree_test(
        X, y, 
        X_test, y_test,
        meta,
        #discretize = False,
        max_depth        = 10 ,
        diffprivacy_mech = "lap",
        budget           = 0.1, 
        criterion        = "entropy", 
        max_candid_features = 70,
        min_samples_leaf = 1,
        print_tree = False,
        is_prune = True,
        debug = False,
        random_state = 1024,
        output_file  = None)

    # redirect output to file
    sys.stdout = open(output_file, 'w')

    print "diffprivacy\t", diffprivacy_mech
    print "budget\t\t", budget
    #print "discretize\t", discretize
    print "max_depth\t", max_depth
    print "max_ftures\t", max_candid_features
    print "criterion\t", criterion
    print "is prune\t", is_prune
    #print "print_tree\t", print_tree
    #print "debug\t\t", debug

    nbtree = NBTreeClassifier(
                max_depth       = max_depth, 
                diffprivacy_mech= diffprivacy_mech, 
                criterion       = criterion, 
                budget          = budget, 
                print_tree      = print_tree, 
                max_candid_features = max_candid_features,
                min_samples_leaf= min_samples_leaf,
                is_prune        = is_prune,
                random_state    = random_state,
                debug   = debug)

    nbtree.set_meta(meta)

    tree = DecisionTreeClassifier(max_depth=max_depth)

    print "fitting...",
    t1 = time()
    #tree.fit(X,y)
    #clf =tree
    nbtree = nbtree.fit(X, y )
    clf = nbtree
    t2 = time()
    print "%.2fs"%(t2-t1)

    y_true = y_test
    y_prob = clf.predict_proba(X_test)[:,-1]
    y_pred = clf.predict( X_test )

    score  = metrics.accuracy_score(    y_test, y_pred)
    matrix = metrics.confusion_matrix(  y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred, 
                        target_names=["label 0", "label 1"])

    print "Accuracy:", score
    print "Matrix:" 
    print matrix

    print "Report:"
    print report

    # sort 
    sorted_indices = np.argsort(- y_prob)
    sorted_y_true = y_true[sorted_indices]
    sorted_y_pred = y_pred[sorted_indices]
    sorted_y_prob = y_prob[sorted_indices]

    auc = metrics.roc_auc_score( y_true, y_prob)
    print "AUC:", auc

    n_samples  = X.shape[0]
    first_list = [] 
    for i in range(1,10):
        first_list.append( i * (n_samples // 10) )

    print 'print the limited number result:'
    print 'first\trecall\tpricsn\tf1\tauc'
    for i in first_list:
            sorted_y_pred = np.zeros(sorted_y_true.size)
            sorted_y_pred[0:i] = 1

            recall    = metrics.recall_score(   sorted_y_true,sorted_y_pred,average='micro')
            precision = metrics.precision_score(sorted_y_true,sorted_y_pred,average='micro')
            #f1_score = f1_score(test_label, predict2, average='micro')
            f1_score=2*precision*recall/(precision+recall)

            print('[%3dK]\t%.3f\t%.3f\t%.3f\t%.3f'%(
                        i//1000, recall, precision, f1_score, auc))
    print "\n"

    print 'print the threshold value result:'
    print 'thresh\trecall\tprecsn\tf1\tauc'
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            y_pred = np.zeros( y_true.size)
            y_pred[np.where( y_prob >= t)] = 1

            recall    = metrics.recall_score(   y_true, y_pred, average='micro')
            precision = metrics.precision_score(y_true, y_pred, average='micro')
            #f1_score = f1_score(y_true, predict2, average='micro')
            f1_score  =2*precision*recall/(precision+recall)
            print('[%.2f]\t%.3f\t%.3f\t%.3f\t%.3f'%(t, recall, precision, f1_score, auc))

def get_feature_importances():
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



def main( data,  is_discretized )

    # criterion
    criterions = ["entropy", "gini"]
    criterion = criterions[0]

    # budgets
    budgets = [ 0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0]

    # feature_importances
    feature_importances = get_feature_importances()

    # features: 70
    n_features = [  70, 60, 50, 40, 30, 20,  
                    17, 14, 11,  9,  7,  5 ]

    # samples
    n_samples  = [  2000000, # 2.0M
                    1500000, # 1.5M
                    1000000, # 1.0M
                     500000, # 0.5M
                     100000, # 0.1M
                      50000, # 50K
                      10000 ]# 10K

    # discretized
    data = preprocessor.load() # discretized


    mech = "no"
    for s in n_samples:
        data = first_n_samples()
        for f in n_features:
            data = first_n_features()
        
            ofile = "d-m_%s-f_%d-s_%dK_d-%d"%(mech,f, s//1000, max_depth)

            nbtree_test( data, 
                         diffprivacy_mech = mech,
                         budget           = -1.0,

                         max_depth        = max_depth,
                         is_prune         = is_prune,
                         output_file      = output_file)

    mechs = ["lap", "exp"]
    for mech in mechs:
        for s in n_samples:
            for f in n_features:
                for b in budgets:
                    data = top(s, f) 
         
                    ofile = "d-m_%s-f_%d-s_%dK-b_%.1f-d-%d"%(mech,f, s//1000, b, max_depth)
            
                    nbtree_test( data, 
                                 diffprivacy_mech = mech,
                                 budget           = b,

                                 max_depth        = max_depth,
                                 is_prune         = is_prune,
                                 output_file      = output_file)

    # not discretized
    data = ... # not discretized

    mech = "no"
    for s in n_samples:
        for f in n_features:
            data = top(s, f) 
            
            nbtree_test( data, 
                         diffprivacy_mech = mech,
                         budget           = -1.0,

                         max_depth        = max_depth,
                         is_prune         = is_prune,
                         output_file      = output_file)

    mech = "exp"
    for s in n_samples:
        for f in n_features:
            for b in budgets:
                data = top(s, f) 
                
                nbtree_test( data, 
                             diffprivacy_mech = mech,
                             budget           = b,

                             max_depth        = max_depth,
                             is_prune         = is_prune,
                             output_file      = output_file)


