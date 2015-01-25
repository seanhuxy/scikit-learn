import os
import sys
import numpy as np
import time
CUR_WORK_DIR= os.getcwd()
sys.path.append(CUR_WORK_DIR)

from sklearn.ensemble.forest import ForestClassifier
from sklearn.tree import NBTreeClassifier

from final_test import preprocess, evaluate

class DPRandomForestClassifier(ForestClassifier):

    def __init__(
                self, 
                base_estimator = NBTreeClassifier,
                n_estimators   = 10,
                bootstrap      = True,
                oob_score      = False,
                n_jobs         = 1,
                random_state   = None,
                verbose        = False,

                # estimator params
                diffprivacy_mech= "no",
                budget          = -1.0,
                criterion       = "gini",
                max_depth       = 10,
                max_features    =100,
                min_samples_leaf= 1,
                is_prune        = True,
                print_tree      = True,
                debug           = False,

                meta =  None
                ):

        super( DPRandomForestClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=(
                            "diffprivacy_mech", "budget", "criterion", 
                            "max_depth", "max_features", "min_samples_leaf", 
                            "is_prune", "print_tree", "debug",
                            "meta"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.diffprivacy_mech = diffprivacy_mech
        if budget > 0.0 and n_estimators > 0:
            budget /= n_estimators
        self.budget =budget

        self.criterion  = criterion
        self.max_depth  = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.is_prune   = is_prune
        self.print_tree = print_tree
        self.debug      = debug

        self.meta = meta

def randomforest_test():

    is_load_from_raw = False
    is_discretize = True

    c_prep = preprocess(is_load_from_raw, is_discretize = is_discretize, dataset="adult")

    p = c_prep

    X, y = p.get_train_X_y()
    X_test, y_test = p.get_test_X_y()
    meta = p.get_features()

    n_estimators = 10

    diffprivacy_mech = "exp"
    budget = 10.0
    criterion = "gini"
    max_depth = 5
    max_features = 14
    min_samples_leaf = 1
    is_prune  = True
    random_state = 4
    print_tree   = False
    debug = False

    nbtree = NBTreeClassifier()
    nbtree.set_meta(meta)

    rf = DPRandomForestClassifier( 
            base_estimator  = nbtree,
            n_estimators    = n_estimators,
            n_jobs          = -1,
            
            diffprivacy_mech= diffprivacy_mech,
            budget          = budget,
            criterion       = criterion,
            max_depth       = max_depth,
            max_features    = max_features,
            min_samples_leaf= min_samples_leaf,
            is_prune        = is_prune,
            random_state    = random_state,
            print_tree      = print_tree,
            debug           = debug,

            meta            = meta)

    rf.fit( X, y )

    evaluate( rf, X_test, y_test)

randomforest_test()


