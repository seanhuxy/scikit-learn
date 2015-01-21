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


def nbtree_test(
        X, y, 
        X_test, y_test,
        meta,

        #discretize = False,
        max_depth = 10 ,
        diffprivacy_mech = "lap",
        budget = 4., 
        criterion="gini", 
        max_candid_features = 70,
        min_samples_leaf = 1,
        print_tree = False,
        is_prune = True,
        debug = False,
        random_state = 1024):

    #print "# ==================================="
    print "diffprivacy\t", diffprivacy_mech
    print "budget\t\t", budget
    #print "discretize\t", discretize
    print "max_depth\t", max_depth
    print "max_features\t", max_candid_features
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

    print "fitting..."
    t1 = time()
    #tree.fit(X,y)
    #clf =tree

    nbtree = nbtree.fit(X, y )
    clf = nbtree

    t2 = time()
    print "Time for fitting %.2fs"%(t2-t1)

    y_true = y_test
    y_prob = clf.predict_proba(X_test)[:,-1]
    y_pred = clf.predict( X_test )

    score  = metrics.accuracy_score(    y_test, y_pred)
    matrix = metrics.confusion_matrix(  y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred, target_names=["label 0", "label 1"])

    print "Accuracy:", score
    print "Matrix:" 
    print matrix

    print "Report:"
    print report

    print "Feature Importance:"
    print "index\tfeature\t\tscore"
    feature_importances = clf.feature_importances_
    
    features = np.argsort(feature_importances)
    for i, f in enumerate(features):
        print "[%2d]\t%25s\t%.3f"%(i, meta.features[f].name, feature_importances[f])
    print "\n"
    
    # sort 
    sorted_indices = np.argsort(- y_prob)
    sorted_y_true = y_true[sorted_indices]
    sorted_y_pred = y_pred[sorted_indices]
    sorted_y_prob = y_prob[sorted_indices]

#    fpr, tpr, threshs = metrics.roc_curve(y_true, y_prob, pos_label=1) 
#    print "thresh\tFPR\tTPR\t"
#    for i, thresh in enumerate(threshs):
#        print "%.2f\t%.2f\t%.2f"%(thresh, fpr[i], tpr[i])

    if False:
        from matplotlib import pyplot as plt 

        plt.plot( fpr, tpr)
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()

    #print y_true
    #print y_prob

    auc = metrics.roc_auc_score( y_true, y_prob)
    print "AUC:", auc

    print 'print the limited number result:'
    print 'first\trecall\tpricsn\tf1\tauc'
    for i in [200,500,800,1100,1500]:
            sorted_y_pred = np.zeros(sorted_y_true.size)
            sorted_y_pred[0:i] = 1

            #auc       = metrics.roc_auc_score(  sorted_y_true, sorted_y_prob)
            recall    = metrics.recall_score(   sorted_y_true, sorted_y_pred, average='micro')
            precision = metrics.precision_score(sorted_y_true, sorted_y_pred, average='micro')
            #f1_score = f1_score(test_label, predict2, average='micro')
            f1_score=2*precision*recall/(precision+recall)

            print('[%d]\t%.3f\t%.3f\t%.3f\t%.3f'%(i, recall, precision, f1_score, auc))
    print "\n"

    print 'print the threshold value result:'
    print 'thresh\trecall\tprecsn\tf1\tauc'
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            y_pred = np.zeros( y_true.size)
            y_pred[np.where( y_prob >= t)] = 1

            #auc       = metrics.roc_auc_score(  y_true, y_prob)
            recall    = metrics.recall_score(   y_true, y_pred, average='micro')
            precision = metrics.precision_score(y_true, y_pred, average='micro')
            #f1_score = f1_score(y_true, predict2, average='micro')
            f1_score  =2*precision*recall/(precision+recall)
            print('[%.2f]\t%.3f\t%.3f\t%.3f\t%.3f'%(t, recall, precision, f1_score, auc))


def preprocess():
    is_load_from_raw = True

    feature_in     = os.path.join(cwd, "dataset/feature.in")
    feature_out    = os.path.join(cwd, "dataset/feature.out")

    #train_data_in  = os.path.join(cwd, "dataset/0506/05_cln.npy")
    train_data_in  = os.path.join(cwd, "dataset/adult.data")
    train_data_out = os.path.join(cwd, "dataset/data.out.npy")

    #test_data_in   = os.path.join(cwd, "dataset/0506/06_cln.npy")
    test_data_in   = os.path.join(cwd, "dataset/adult.test")
    test_data_out  = os.path.join(cwd, "dataset/test.out.npy")

    #test_data_in = None

    preprocessor = Preprocessor()
    if is_load_from_raw:
        preprocessor.load( feature_in, train_data_in, test_data_in, sep=" ", 
                            is_discretize= True, nbins=10)

        preprocessor.export( feature_out, train_data_out, test_data_out)

    else:
        preprocessor.load_existed(feature_out, train_data_out, test_data_out)

    return preprocessor

if __name__ == "__main__":

    preprocessor = preprocess()
    #exit()

    train = preprocessor.get_train()
    test  = preprocessor.get_test()

    X,      y      = train[:,:-1], train[:,-1]
    X_test, y_test = test[:,:-1],  test[:,-1]

    y = np.ascontiguousarray(y)
    y_test = np.ascontiguousarray(y_test)

    t1 = time()
    nbtree_test(X, y, X_test, y_test, preprocessor)
    t2 = time()
    print "Time costs %.2fs"%(t2-t1)


