"""
A test python script, to compare the accuracy, auc 
of three different differential privacy classifications
in different setting of the number of features
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
OUTPUT_DIR  = os.path.join(CUR_WORK_DIR, "log/feature")
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
N_FEATURES  = [  4, 8, 12, 16, 20, 30, 40, 50, 60, 70 ]          
N_SAMPLES   = [  100000, # 0.1M
                1000000, # 1.0M 
              ]
# XXX
N_FEATURES = [ 3, 7, 10, 14]
N_SAMPLES  = [ 20000]

def test(cp, dp, feature_importances, 
         n_samples,
         budget
         ):

    auc_str = StringIO()
    scr_str = StringIO()

    stat_name = "%dKs_%.2fb.stat"%(n_samples//1000, budget)
    stat_path = join(OUTPUT_DIR, stat_name)

    log_name = "%dKs_%.2fb.log"%(n_samples//1000, budget)
    log_path = join(OUTPUT_DIR, log_name)
    open(log_path, 'w').close()

    auc_str.write("AUC table -- %dKs, %.2fb\n"\
                %(n_samples//1000, budget))
    scr_str.write("Score table -- %dKs, %.2fb\n"\
                %(n_samples//1000, budget))


    auc_str.write("clf\tdis\t")
    scr_str.write("clf\tdis\t")
    for f in N_FEATURES:
        auc_str.write("%2d\t"%f)
        scr_str.write("%2d\t"%f)
    scr_str.write("\n")
    auc_str.write("\n")

    for mech in MECHS:
        for disc in DISCS:
            if mech is "lap" and disc is False:
                continue

            p = dp if disc else cp
            auc_str.write("%s\t%c\t"%(mech, DCHAR[disc]))
            scr_str.write("%s\t%c\t"%(mech, DCHAR[disc]))

            for f in N_FEATURES:
                
                X, y, X_t, y_t, meta = p.get_first_nsf(
                                            n_samples,
                                            f,
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
    for s in N_SAMPLES:
        for b in BUDGETS:
            jobs.append( 
                delayed(test)(cp, dp,
                                feature_importances,
                                s, b) 
            )

    Parallel(n_jobs = N_CORES, max_nbytes=1e3)( jobs )

main()
