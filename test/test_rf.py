from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import ForestClassifier

class DiffPrivacyRandomForestClassifier(ForestClassifier):

    def __init__(self, 
                base_estimator = NBTreeClassifier,
                n_estimators   = 10,
                bootstrap=True,
                oob_score=False,
                n_jobs=1,
                random_state=None,
                verbose=False,
                # estimator params
                diffprivacy_mech = "no",
                budget = -1.0,
                criterion = "gini",
                max_depth = 10,
                min_samples_leaf = 1,
                is_prune = True,
                print_tree = True,
                debug=False,
                ):

        super( DiffPrivacyRandomForestClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=("diffprivacy_mech","budget","criterion", 
                            "max_depth", "min_samples_leaf", "is_prune",
                            "print_tree", "debug"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.diffprivacy_mech=diffprivacy_mech
        if budget > 0.0 and n_estimators > 0:
            budget /= n_estimators
        self.budget=budget

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.is_prune = is_prune
        self.print_tree = True
        self.debug = debug

    n_estimators = 10

    rf = DiffPrivacyRandomForestClassifier( 
            base_estimator=nbtree,
            n_estimators = n_estimators,
            n_jobs=-1,
            
            diffprivacy_mech=diffprivacy_mech,
            budget=budget,
            criterion    = criterion,
            max_depth    = max_depth,
            min_samples_leaf=min_samples_leaf,
            is_prune = is_prune,
            print_tree=print_tree,
            debug=debug)
    #rf.fit( X, y )


