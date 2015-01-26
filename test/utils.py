
DATA_DIR = os.path.join(CUR_WORK_DIR, "dataset")
FEATURE_IMPORTANCE_FILE = os.path.join(CUR_WORK_DIR,
                            "dataset/feature_importance.npy"

def build(
        X, y, meta,

        is_discretize    = False,
        diffprivacy_mech = "lap",
        budget           = 0.1, 

        criterion        = "entropy", 
        max_depth        = 10,
        max_features     = 70,
        min_samples_leaf = 1,
        is_prune         = True,

        print_tree       = False,
        debug            = False,
        random_state     = 1000,
        output_file      = None):

    # redirect output to file
    if output_file is None:
        sys.stdout = sys.__stdout__
    else:
        sys.stdout = open(output_file, 'a')
    
    print "---------------------------------------------"
    if is_discretize:
        dchar = 'd'
    else:
        dchar = 'c'
    print "samples\tfeatures\tmech\tdisc\tbudget"
    print "%6dK\t%8d\t%4s\t%4s\t%6.2f"%(
            X.shape[0]//1000, X.shape[1], 
            diffprivacy_mech, dchar, budget)
    print "---------------------------------------------"

    #print "mech\t",  diffprivacy_mech
    #print "budget\t\t",     budget
    #print "discretize\t",   is_discretize
    #print "max_depth\t",    max_depth
    #print "max_ftures\t",   max_features
    #print "criterion\t",    criterion
    #print "is prune\t",     is_prune
    #print "output\t\t",       output_file
    #print "print_tree\t",  print_tree
    #print "debug\t\t",     debug

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
    print "fitting costs\t%.2fs"%(t2-t1)

    return clf


    pass


def evaluate( clf, X_test, y_test):

    y_true = y_test
    y_prob = clf.predict_proba(X_test)[:,-1]
    y_pred = clf.predict( X_test )

    score  = metrics.accuracy_score(    y_test, y_pred)
    auc    = metrics.roc_auc_score(     y_true, y_prob)
    matrix = metrics.confusion_matrix(  y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred, 
                        target_names=["label 0", "label 1"])

    print "Score:\t%.5f"%score
    print "AUC:\t%.5f"%auc

    print "Matrix"
    print matrix

    print "Report:"
    print report

    # sort 
    sorted_indices = np.argsort(- y_prob)
    sorted_y_true = y_true[sorted_indices]
    sorted_y_pred = y_pred[sorted_indices]
    sorted_y_prob = y_prob[sorted_indices]

    n_samples  = X_test.shape[0]
    first_list = [] 
    for i in range(1,10):
        first_list.append( i * (n_samples // 10) )

    print 'limited number result:'
    print 'first\trecall\tpricsn\tf1  \tauc'
    for i in first_list:
            sorted_y_pred = np.zeros(sorted_y_true.size)
            sorted_y_pred[0:i] = 1

            recall    = metrics.recall_score(   
                            sorted_y_true,sorted_y_pred,
                            average='micro')
            precision = metrics.precision_score(
                            sorted_y_true,sorted_y_pred,
                            average='micro')
            #f1_score = f1_score(test_label, predict2, average='micro')
            f1_score=2*precision*recall/(precision+recall)

            print('%3dK\t%.3f\t%.3f\t%.3f\t%.3f'%(
                        i//1000, recall, precision, f1_score, auc))
    print "\n"

    print 'threshold value result:'
    print 'thresh\trecall\tprecsn\tf1  \tauc'
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            y_pred = np.zeros( y_true.size)
            y_pred[np.where( y_prob >= t)] = 1

            recall    = metrics.recall_score(   y_true, y_pred, 
                                                average='micro')
            precision = metrics.precision_score(y_true, y_pred, 
                                                average='micro')
            #f1_score = f1_score(y_true, predict2, average='micro')
            f1_score  =2*precision*recall/(precision+recall)
            print('%.2f\t%.3f\t%.3f\t%.3f\t%.3f'%
                    (t, recall, precision, f1_score, auc))

    return score, auc

def load_feature_importances():
    filename = FEATURE_IMPORTANCE_FILE 
    features = np.load(filename)
    return features

def cal_feature_importances(X, y, meta):

    print "get feature importance.."

    filename = FEATURE_IMPORTANCE_FILE 
    features = np.load(filename)
    return features

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

    np.save(  filename, features )

    return features

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


def get_data(
            is_load_from_raw=False, 
            is_discretize=False, 
            dataset="liantong", 
            dmethod="cluster"):

    feature_in     = os.path.join(DATA_DIR, "feature.in")
    if dataset == "liantong":
        train_data_in  = os.path.join(DATA_DIR, "0506/05_cln.npy")
        test_data_in   = os.path.join(DATA_DIR, "0506/06_cln.npy")
    else:
        train_data_in  = os.path.join(DATA_DIR, "adult.data")
        test_data_in   = os.path.join(DATA_DIR, "adult.test")

    if is_discretize: 
        feature_out    = os.path.join(DATA_DIR, "feature_d.out")
        train_data_out = os.path.join(DATA_DIR, "data_d.out.npy")
        test_data_out  = os.path.join(DATA_DIR, "test_d.out.npy")
    else:
        feature_out    = os.path.join(DATA_DIR, "feature_c.out") 
        train_data_out = os.path.join(DATA_DIR, "data_c.out.npy")
        test_data_out  = os.path.join(DATA_DIR, "test_c.out.npy")


    preprocessor = Preprocessor()
    if is_load_from_raw:
        preprocessor.load( feature_in, train_data_in, test_data_in, 
                            is_discretize = is_discretize, nbins=10, 
                            dmethod=dmethod)
        preprocessor.export( feature_out, 
                            train_data_out, test_data_out)

    else:
        preprocessor.load_existed(feature_out, 
                        train_data_out, test_data_out)
    return preprocessor

