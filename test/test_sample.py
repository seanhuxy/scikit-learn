import os
from os.path import join
sys.path.append(CUR_WORK_DIR)
CUR_WORK_DIR= os.getcwd()
OUTPUT_DIR  = os.path.join(CUR_WORK_DIR, "log/sample")

from cStringIO import StringIO

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
data = ...
output_dir = "log/sample_test"

n_features = [ 20, 50 ]                
n_samples  = [    10000, # 10K
                  50000, # 50K
                 100000, # 0.1M 
                 500000, # 0.5M
                1000000, # 1.0M
                1500000, # 1.5M
                2000000, # 2.0M
            ]
budgets = [ 0.01, 0.1, 0.5, 1., 3., 5., 7., 9.]
mechs = ["no", "lap", "exp"]
discs = [ True, False ]

def test(cp, dp, feature_importances, 
         n_features,
         budget
         ):

    auc_str = StringIO()
    src_str = StringIO()

    stat_name = "%df_%.2fb.stat"%(n_features, budget))
    stat_path = join(OUTPUT_DIR, log_name)

    log_name = "%df_%.2fb.log"%(n_features, budget))
    log_path = join(OUTPUT_DIR, log_name)
    open(log_path, 'w').close()

    auc_str.write("AUC table -- %d f %.2f b\n"%(\
                    n_features, budget))
    scr_str.write("Score table -- %d f %.2f b\n"%
                    (n_features, budget))

    auc_str.write("clf\tdis\t")
    scr_str.write("clf\tdis\t")
    for s in n_samples:
        auc_str.write("%4dK\t"%s//1000)
        scr_str.write("%4dK\t"%s//1000)
    scr_str.write("\n")
    auc_str.write("\n")

    for mech in mechs:
        for disc in discs:
            if mech is "lap" and disc is False:
                continue

            auc_str.write("%s\t%c\t"%(mech, dchar))
            scr_str.write("%s\t%c\t"%(mech, dchar))

            for s in n_samples:
                
                p = dp if disc else cp

                X, y, X_t, y_t, meta = p.get_first_nsf(
                                            s,
                                            n_features,
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
    for f in n_features:
        for b in budgets:
            jobs.append( 
                delayed(test)(cp, dp,
                                feature_importances,
                                f,
                                budget = b) 
            )

    n_cores = 30
    Parallel(n_jobs = n_cores, max_nbytes=1e3)( jobs )


