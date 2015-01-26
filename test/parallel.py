import os
import cStringIO
import sys
import numpy as np
from time import time

CUR_WORK_DIR= os.getcwd()
sys.path.append(CUR_WORK_DIR)

from sklearn.externals.joblib import Parallel, delayed

from final_test import nbtree_test, evaluate, preprocess

def get_first_nsf( data, features, n_train, n_test, 
                    n_samples, n_features, feature_importances):
      
    #print "get first %dK smpl and %d f"%(n_samples//1000, n_features)
    t0 = time()
    features = features[ feature_importances[ : n_features] ]

    train = data[         : n_train, :]
    X = train[ : n_samples, feature_importances[ : n_features]]
    y = train[ : n_samples, -1]

    test  = data[ n_train : n_train + n_test, : ]
    X_test = test[ : , feature_importances[ : n_features]]
    y_test = test[ : , -1]

    t1 = time()
    #print "cost %.2fs"%(t1-t0)

    return X, y, X_test, y_test, features


def loop(
        c_prep, d_prep,
         feature_importances,
         n_samples, n_features
         ):

    pid = os.getpid()

    #is_load_from_raw = False
    #c_prep = preprocess(is_load_from_raw, is_discretize = False)
    #d_prep = preprocess(is_load_from_raw, is_discretize = True)

    criterion = "gini"
    is_prune = True
    max_depth = 5
    
    OUTPUT_DIR = os.path.join(os.getcwd(), "para_log")  

    # budgets 8
    budgets = [ 0.01, 0.1, 0.5, 1., 3., 5., 7., 9.]
    #budgets = [ 9., 5., 1., 0.5, ]

    log_fname = os.path.join(OUTPUT_DIR, 
                "%dKs_%df.log"%(n_samples//1000, n_features))

    stat_fname = os.path.join(OUTPUT_DIR, 
                "%dKs_%df.stat"%(n_samples//1000, n_features))

    # clear the log file
    open(log_fname, 'w').close()

    auc_file = cStringIO.StringIO()
    scr_file = cStringIO.StringIO()
    auc_file.write("AUC table -- %dK samples, %d features\n"%(\
                    n_samples//1000, n_features))

    auc_file.write("clf\tdis\t")
    for b in budgets:
        auc_file.write("b=%.2f\t"%b)
    auc_file.write("\n")

    scr_file.write("Score table -- %dK samples, %d features\n"%
                    (n_samples//1000, n_features))
    scr_file.write("clf\tdis\t")
    for b in budgets:
        scr_file.write("b=%.2f\t"%b)
    scr_file.write("\n")
    
    cnt = 0
    for mech in ["org", "no", "lap", "exp"]:
        for is_discretize in [ False, True ]:
            if mech is "lap" and is_discretize is False:
                continue

            if is_discretize:
                p = d_prep
                dchar = 'd'
            else:
                p = c_prep
                dchar = 'c'

            data     = p.data
            features = p.features
            n_train  = p.n_train_samples
            n_test   = p.n_test_samples

            X, y, X_test, y_test, meta = \
                get_first_nsf(data, features, n_train, n_test, 
                            n_samples, n_features, 
                            feature_importances )

            auc_file.write("%s\t%c\t"%(mech, dchar))
            scr_file.write("%s\t%c\t"%(mech, dchar))

            first = True
            for b in budgets:

                if not first and mech in ["org", "no"]:
                    pass
                else:
                    clf = nbtree_test(X, y, X_test, y_test, meta,
                                     diffprivacy_mech = mech,
                                     budget           = b,

                                     is_discretize    = is_discretize,

                                     max_depth        = max_depth,
                                     is_prune         = is_prune,
                                     output_file      = log_fname)

                    score, auc = evaluate(clf, X_test, y_test)

                auc_file.write("%.2f\t"%(auc*100.))
                scr_file.write("%.2f\t"%(score*100.))

                sys.stdout = sys.__stdout__
                cnt += 1
                #print "[%5d][%3d]%6dK\t%8d\t%4s\t%4s\t%6.2f\t%5.2f\t%5.2f"%(
                    #pid, cnt, n_samples//1000, n_features, 
                    #mech, dchar, b, auc*100., score*100.)

                if first:
                    first = False

            auc_file.write("\n")
            scr_file.write("\n")
   
    sys.stdout = sys.__stdout__
    print "------------------------------------------------------"
    print auc_file.getvalue()
    print scr_file.getvalue()

    sf = open(stat_fname, 'w')
    sf.write( auc_file.getvalue())
    sf.write( scr_file.getvalue())
    sf.close()

if __name__ == "__main__":
 
    is_load_from_raw = True
    dmethod = "cluster"
    c_prep = preprocess(is_load_from_raw, is_discretize = False, 
                        dataset="adult")
    d_prep = preprocess(is_load_from_raw, is_discretize = True, 
                        dataset="adult",
                        dmethod = dmethod)

    filename = os.path.join( os.getcwd(), 
            "dataset/feature_importance.npy")
    feature_importances = np.load(filename)

    n_features = [  5, 7, 9, 11, 14, 17, 20,  30, 40, 50, 60, 70 ]
    n_samples  = [    10000, # 10K
                      50000, # 50K
                     100000, # 0.1M
                     500000, # 0.5M
                    1000000, # 1.0M
                    1500000, # 1.5M
                    2000000, # 2.0M
                ]


    # sample 5X2
    n_features = [ 20, 50 ]                
    n_samples  = [    10000, # 10K
                      50000, # 50K
                   #  100000, # 0.1M 
                     500000, # 0.5M
                   # 1000000, # 1.0M
                    1500000, # 1.5M
                    2000000, # 2.0M
                ]

    n_features = [  14]
    n_samples  = [  20000]

    jobs = []
    for s in n_samples:
        for f in n_features:
            jobs.append( 
                delayed( loop )(c_prep, d_prep,
                                feature_importances,
                                s, f) 
            )

    # feature 10X2
    n_features = [  4, 8, 12, 16, 20, 30, 40, 50, 60, 70 ]          
    n_samples  = [  100000, # 0.1M
                    1000000, # 1.0M
                ]

    n_features = [  ]
    n_samples  = [  ]
    for s in n_samples:
        for f in n_features:
            jobs.append( 
                delayed( loop )(c_prep, d_prep,
                                feature_importances,
                                s, f) 
            )

    n_cores = 1
    n_trees = len(jobs) * 28
    print "number of jobs %d"%(len(jobs))
    print "number of tree %d"%(n_trees)
    print "estimated time %.2f mins per core"%( 3. * n_trees / n_cores)

    print "[ pid ][cnt]samples\tfeatures\tmech\tdisc\tbudget\t auc \tscore"
    Parallel(n_jobs = n_cores, max_nbytes=1e3)( jobs )


