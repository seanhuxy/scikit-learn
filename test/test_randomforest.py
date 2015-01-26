import os
from os.path import join

CUR_WORK_DIR= os.getcwd()
OUTPUT_DIR  = os.path.join(CUR_WORK_DIR, "log/randomforest")
sys.path.append(CUR_WORK_DIR)

from cStringIO import StringIO

import sys
import numpy as np
import time

from sklearn.ensemble.forest import ForestClassifier
from sklearn.tree import NBTreeClassifier


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


# constant
dmethod = "cluster"
criterion = "gini"
is_prune  = True
max_features = 70

# Input and Output
output_dir = "log/sample_test"

max_depths = [ 3, 5, 10]
n_features = [ 50 ]          
n_samples  = [ 1000000, # 1.0M ]

# budgets 
budgets = [  0.5, 1., 3., 5., 7., 9.]
mechs = ["org", "no", "lap", "exp"]
discs = [ True, False ]

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


# constant
# get data
is_load_from_raw = False
is_discretize = False
dataset = "liantong"
dmethod = "cluster"

criterion = "gini"
is_prune  = True
max_depth = 10
max_features = 70

# Input and Output
output_dir = "log/test_randomforest"

# Variable
n_features = [  50  ]          
n_samples  = [ 1000000, # 1.0M ]

budgets = [ 0.01, 0.1, 0.5, 1., 3., 5., 7., 9.]
mechs = ["no", "lap", "exp"]
discs = [ True, False ]

def test(cp, dp, feature_importances, 
         n_samples,
         budget
         ):

    auc_str = StringIO()
    src_str = StringIO()

    stat_name = "%dKs_%.2fb.stat"%(n_samples//1000, budget))
    stat_path = join(OUTPUT_DIR, log_name)

    log_name = "%dKs_%.2fb.log"%(n_samples//1000, budget))
    log_path = join(OUTPUT_DIR, log_name)
    open(log_path, 'w').close()

    auc_str.write("AUC table -- %dK s %.2f b\n"%(\
                    n_samples//1000, budget))
    scr_str.write("Score table -- %dK s %.2f b\n"%
                    (n_samples//1000, budget))

    auc_str.write("clf\tdis\t")
    scr_str.write("clf\tdis\t")
    for f in n_features:
        auc_str.write("%2d\t"%f)
        scr_str.write("%2d\t"%f)
    scr_str.write("\n")
    auc_str.write("\n")

    for rf in [True, False]:
        for mech in mechs:
            for disc in discs:

                if mech is "lap" and disc is False:
                    continue

                auc_str.write("%s\t%c\t"%(mech, dchar))
                scr_str.write("%s\t%c\t"%(mech, dchar))
                
                p = dp if disc else cp

                X, y, X_t, y_t, meta = p.get_first_nsf(
                                            n_samples,
                                            f,
                                            feature_importances)

                clf = build(X, y, meta,
                      is_discretize = disc,
                      diffprivacy_mech = mech,
                      budget = budget,

                      criterion = criterion,
                      max_depth = max_depth,
                      min_samples_leaf = min_samples_leaf,
                      is_prune  = is_prune,

                      print_tree = print_tree,
                      debug = debug,
                      random_state = random_state,
                      output_file = log_path)

                score, auc = evaluate( clf, X_t, y_t, log_path)

                auc_str.write("%.2f\t"%(auc*100.))
                scr_str.write("%.2f\t"%(score*100.))

    sys.stdout = sys.__stdout__
    print "------------------------------------------------------"
    print auc_str.getvalue()
    print scr_str.getvalue()
    
    sf = open(stat_path, 'w')
    sf.write( auc_str.getvalue())
    sf.write( scr_str.getvalue())
    sf.close()

def main():

    cp = get_data(False, False)
    dp = get_data(False, True )
    feature_importances = load_feature_importances()

    jobs = []
    for s in n_samples:
        for b in budgets:
            jobs.append( 
                delayed(test)(cp, dp,
                                feature_importances,
                                s,
                                budget = b) 
            )

    n_cores = 30
    Parallel(n_jobs = n_cores, max_nbytes=1e3)( jobs )


