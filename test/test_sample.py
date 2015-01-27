"""
A test python script, to compare the accuracy, auc 
of three different differential privacy classifications
in different setting of the size of train dataset
"""
__author__ = "Xueyang Hu"
__email__  = "huxuyangs@gmail.com"

import os
import sys
from os.path import join
from cStringIO import StringIO
CUR_WORK_DIR= os.getcwd() # Current Work Directory
sys.path.append(CUR_WORK_DIR)
from sklearn.externals.joblib import Parallel, delayed
from utils import *

# Constant for Input and Output
LOAD_FROM_TXT = False   # load dataset from txt or binary
DATASET     = "liantong"# data set
DISC_METHOD = "cluster" # the method for discretizing continuous 
                        # features

# output directory for logs
OUTPUT_DIR  = os.path.join(CUR_WORK_DIR, "log/sample")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Number of Cores for parallel computing
N_CORES = 30

# Constant Parameters For Building Tree
CRITERION = "gini"
IS_PRUNE  = True
MAX_DEPTH = 10
MAX_FEATURES = 70
MIN_SAMPLES_LEAF = 1

PRINT_TREE = False
DEBUG      = False

RANDOM_STATE = 1

MECHS = ["no", "lap", "exp"]
DISCS = [ True, False ]
DCHAR = { True  : 'd', 
          False : 'c' }

BUDGETS = [ 0.01, 0.1, 0.5, 1., 3., 5., 7., 9.]

# Variables
N_FEATURES = [ 20, 50 ]                
N_SAMPLES  = [    10000, # 10K
                  50000, # 50K
                 100000, # 0.1M 
                 500000, # 0.5M
                1000000, # 1.0M
                1500000, # 1.5M
                2000000, # 2.0M
            ]

# XXX
N_FEATURES = [14]
N_SAMPLES  = [ 5000, 10000, 20000]

def test(cp, dp, feature_importances, 
         n_features,
         budget
         ):

    auc_str = StringIO()
    scr_str = StringIO()

    stat_name = "%df_%.2fb.stat"%(n_features, budget)
    stat_path = join(OUTPUT_DIR, stat_name)

    log_name = "%df_%.2fb.log"%(n_features, budget)
    log_path = join(OUTPUT_DIR, log_name)
    open(log_path, 'w').close()

    auc_str.write("AUC table -- %df, %.2fb\n"\
                %(n_features, budget))
    scr_str.write("Score table -- %df, %.2fb\n"\
                %(n_features, budget))

    auc_str.write("clf\tdis\t")
    scr_str.write("clf\tdis\t")
    for s in N_SAMPLES:
        auc_str.write("%4dK\t"%(s//1000))
        scr_str.write("%4dK\t"%(s//1000))
    scr_str.write("\n")
    auc_str.write("\n")

    for mech in MECHS:
        for disc in DISCS:
            if mech is "lap" and disc is False:
                continue

            p = dp if disc else cp

            auc_str.write("%s\t%c\t"%(mech, DCHAR[disc]))
            scr_str.write("%s\t%c\t"%(mech, DCHAR[disc]))

            for s in N_SAMPLES:
                

                X, y, X_t, y_t, meta = p.get_first_nsf(
                                            s,
                                            n_features,
                                            feature_importances)
                clf = build(X, y, meta,
                      is_discretize = disc,
                      diffprivacy_mech = mech,
                      budget = budget,

                      criterion = CRITERION,
                      max_depth = MAX_DEPTH,
                      max_features = MAX_FEATURES,
                      min_samples_leaf = MIN_SAMPLES_LEAF,
                      is_prune  = IS_PRUNE,

                      print_tree = PRINT_TREE,
                      debug      = DEBUG,
                      random_state = RANDOM_STATE,
                      output_file = log_path)

                score, auc = evaluate( clf, X_t, y_t, log_path)

                auc_str.write("%.2f\t"%(auc*100.))
                scr_str.write("%.2f\t"%(score*100.))

            auc_str.write("\n")
            scr_str.write("\n")

    sys.stdout = sys.__stdout__
    print "------------------------------------------------------"
    print auc_str.getvalue()
    print scr_str.getvalue()
    
    sf = open(stat_path, 'w')
    sf.write( auc_str.getvalue())
    sf.write( scr_str.getvalue())
    sf.close()

def main():

    cp = get_data(LOAD_FROM_TXT, False)
    dp = get_data(LOAD_FROM_TXT, True )
    feature_importances = load_feature_importances()

    jobs = []
    for f in N_FEATURES:
        for b in BUDGETS:
            jobs.append( 
                delayed(test)(cp, dp,
                            feature_importances, f, b) 
            )

    Parallel(n_jobs = N_CORES, max_nbytes=1e3)( jobs )

main()

