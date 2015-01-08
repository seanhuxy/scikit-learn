import os
import sys
import numpy as np

#abspath = os.path.abspath(__file__)

cwd = os.getcwd()
sys.path.append(cwd)

import sklearn
from sklearn.tree import NBTreeClassifier
from sklearn.cross_validation import cross_val_score

from preprocessor import Preprocessor


def cross_val():

    output =  cross_val_score(nbtree, X, y, cv=5, fit_params={'meta':meta, 'debug':debug})

    print output
    print "Average Accuracy:", np.average(output)
    print "# =========================================" 
    print "\n"

def nbtree_test(
        X, y, 
        X_test, y_test,
        meta,

        discretize = False,
        max_depth = 10,
        diffprivacy_mech = "exp",
        budget =10., 
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

    nbtree = nbtree.fit(X,y,meta, debug = debug)

    y_pred = nbtree.predict( X_test )

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    score  = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred) 
    report = classification_report(y_test, y_pred)

    print "Accuracy:", score
    print "Matrix:" 
    print matrix

    print "Report:"
    print report

    import matplotlib.pyplot as plt
    # Show confusion matrix in a separate window
    plt.matshow(matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def preprocess():
    is_load_from_raw = True

    #feature_in     = os.path.join(cwd, "dataset/adult_nomissing.arff")
    #feature_in    = os.path.join(cwd, "dataset/feature.in")
    feature_in    = os.path.join(cwd, "dataset/adult.feature")


    feature_out    = os.path.join(cwd, "dataset/adult.feature.out")

    train_data_in  = os.path.join(cwd, "dataset/adult_nomissing.data")
    train_data_out = os.path.join(cwd, "dataset/adult.data.out")

    test_data_in   = os.path.join(cwd, "dataset/adult_nomissing.test")
    test_data_out  = os.path.join(cwd, "dataset/adult.test.out")

    preprocessor = Preprocessor()
    if is_load_from_raw:
        preprocessor.load( feature_in, train_data_in, test_data_in, sep=" ")
        #preprocessor.discretize(nbins=10)
        preprocessor.export( feature_out, train_data_out, test_data_out)

    else:
        preprocessor.load_existed(feature_out, train_data_out, test_data_out)

        #for f in preprocessor.features:
        #    print f
        print "Preprocess data"
        print preprocessor.data.dtype
        print preprocessor.data
        #print preprocessor.get_train()
        #print preprocessor.get_test()

    print "end preprocess" 
    return preprocessor

if __name__ == "__main__":

    preprocessor = preprocess()

    train = preprocessor.get_train()
    test  = preprocessor.get_test()

    X,      y      = train[:,:-1], train[:,-1]
    X_test, y_test = test[:,:-1], test[:,-1]

    nbtree_test(X, y, X_test, y_test, preprocessor)


