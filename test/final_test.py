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

import preprocessor
from preprocessor import Preprocessor



def first_n_features( feature_importances, data, n_features ):
    data = data[ : , feature_importances[ :n_features] ]
    return data

def first_n_samples( data, n_samples ):
	
    data = data[ : n_samples]
    return data
    # or bagging

def nbtree_test(
        X, y, X_test, y_test, meta,

        is_discretize    = False,
        max_depth        = 10 ,
        diffprivacy_mech = "lap",
        budget           = 0.1, 
        criterion        = "entropy", 
        max_features     = 70,
        min_samples_leaf = 1,
        print_tree       = False,
        is_prune         = True,
        debug            = False,
        random_state     = 1000,
        output_file      = None):

    # redirect output to file
    if output_file is None:
        sys.stdout = sys.__stdout__
    else:
        sys.stdout = open(output_file, 'w')

    print "train set: %3dK samples, %2d features"%(X.shape[0], X.shape[1])
    print "test  set: %3dK samples, %2d features"%(X_test.shape[0], X_test.shape[1])
    print "diffprivacy\t",  diffprivacy_mech
    print "budget\t\t",     budget
    print "discretize\t",   is_discretize
    print "max_depth\t",    max_depth
    print "max_ftures\t",   max_features
    print "criterion\t",    criterion
    print "is prune\t",     is_prune
    print "output\t\t",       output_file
    #print "print_tree\t",  print_tree
    #print "debug\t\t",     debug

    print "fitting...",
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
    print "%.2fs"%(t2-t1)

    return clf

def evaluate( clf, X_test, y_test):

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

            recall    = metrics.recall_score(   
                            sorted_y_true,sorted_y_pred,average='micro')
            precision = metrics.precision_score(
                            sorted_y_true,sorted_y_pred,average='micro')
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
            print('[%.2f]\t%.3f\t%.3f\t%.3f\t%.3f'%
                    (t, recall, precision, f1_score, auc))
    return score, auc

def get_feature_importances(X, y, meta):

    # get from a standard classifier algor
    tree = DecisionTreeClassifier()

    t1 = time()
    tree.fit( X, y)
    clf = tree
    t2 = time()
    print "Time for fitting %.2fs"%(t2-t1)
    
    feature_importances = clf.feature_importances_
    features = np.argsort( - feature_importances) # by descending order
    print features
    for i, f in enumerate(features):
        print "[%2d]\t%25s\t%.3f"%(f, meta[f].name, feature_importances[f])
    print "\n"

    return features

def preprocess( is_load_from_raw=True, is_discretize=False):

    feature_in     = os.path.join(CUR_WORK_DIR, "dataset/feature.in")
    #feature_out    = os.path.join(CUR_WORK_DIR, "dataset/feature_c.out") #XXX
    feature_out    = os.path.join(CUR_WORK_DIR, "dataset/feature.out") #XXX

    #train_data_in  = os.path.join(CUR_WORK_DIR, "dataset/0506/05_cln.npy")
    train_data_in  = os.path.join(CUR_WORK_DIR, "dataset/adult.data")
    #train_data_out = os.path.join(CUR_WORK_DIR, "dataset/data_c.out.npy")
    train_data_out = os.path.join(CUR_WORK_DIR, "dataset/data.out.npy")


    #test_data_in   = os.path.join(CUR_WORK_DIR, "dataset/0506/06_cln.npy")
    test_data_in   = os.path.join(CUR_WORK_DIR, "dataset/adult.test")
    #test_data_out  = os.path.join(CUR_WORK_DIR, "dataset/test_c.out.npy")
    test_data_out  = os.path.join(CUR_WORK_DIR, "dataset/test.out.npy")


    preprocessor = Preprocessor()
    if is_load_from_raw:
        preprocessor.load( feature_in, train_data_in, test_data_in, sep=" ", 
                            is_discretize= is_discretize, nbins=10)

        preprocessor.export( feature_out, train_data_out, test_data_out)

    else:
        preprocessor.load_existed(feature_out, train_data_out, test_data_out)

    return preprocessor

if __name__ == "__main__":
    is_load_from_raw = True
    is_discretize    = False

    c_prep = preprocess(is_load_from_raw, is_discretize = False)
    d_prep = preprocess(is_load_from_raw, is_discretize = True)

    #for f in c_prep.features:
    #    f.type = preprocessor.FEATURE_CONTINUOUS
    #    f.n_values = 2
    
    # criterion
    criterions = ["entropy", "gini"]
    criterion = criterions[0]

    # budgets
    budgets = [ 9., 7., 5., 3., 1., 0.5, 0.1, 0.01 ]

    # feature_importances
    X, y = c_prep.get_train_X_y()
    feature_importances = get_feature_importances(X, y, c_prep.features)

    # features:
    #n_features = [  70, 60, 50, 40, 30, 20, 17, 14, 11,  9,  7,  5 ]
    n_features = [14, 10, 7]

    # samples
    n_samples  = [  2000000, # 2.0M
                    1500000, # 1.5M
                    1000000, # 1.0M
                     500000, # 0.5M
                     100000, # 0.1M
                      50000, # 50K
                      10000 ]# 10K
    n_samples  = [ 20000, 10000 ]

    max_depth = 10
    is_prune  = True

    cnt = 0

    for s in n_samples:
        #X, y = first_n_samples( __train_data , s )
        for f in n_features:
               
            for is_discretize in [False, True]:

                if is_discretize is True:
                    X, y, X_test, y_test, meta = d_prep.get_first_nsf( s, f, feature_importances )
                    dpmechs = ["lap", "exp"]
                    dchar = 'd'
                    
                else:
                    X, y, X_test, y_test, meta = c_prep.get_first_nsf( s, f, feature_importances )
                    dpmechs = ["exp"]
                    dchar = 'c'

                #dpmechs = []
                
                nodpmechs = ["no", "org"]
                for mech in nodpmechs:

                    filename = "f%02d__s%dk__m%s__%c.log"%(f, s//1000, mech, dchar)
                    output_file = os.path.join(OUTPUT_DIR, filename)
                    
                    clf =nbtree_test(X, y, X_test, y_test, meta,
                                     diffprivacy_mech = mech,
                                     budget           = -1.0,

                                     is_discretize    = is_discretize,

                                     max_depth        = max_depth,
                                     is_prune         = is_prune,
                                     output_file      = output_file)

                    score, auc = evaluate(clf, X_test, y_test)

                    sys.stdout = sys.__stdout__
                    print "[%2d]%-36s: %.4f, %.4f"%(cnt, filename, score, auc)
                    cnt += 1

                for mech in dpmechs:
                    for b in budgets:

                        filename = "f%02d__s%dk__m%s__%c__b%.1f.log"%(f, s//1000, mech, dchar, b )
                        output_file = os.path.join(OUTPUT_DIR, filename)
                       
                        clf =nbtree_test(X, y, X_test, y_test, meta,
                                         diffprivacy_mech = mech,
                                         budget           = b,

                                         max_depth        = max_depth,
                                         is_prune         = is_prune,
                                         output_file      = output_file)

                        score, auc = evaluate(clf, X_test, y_test)

                        sys.stdout = sys.__stdout__
                        print "[%2d]%-36s: %.4f, %.4f"%(cnt, filename, score, auc)
                        cnt += 1
            print ""

