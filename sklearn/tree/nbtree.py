"""
THis module implement no binary tree classifier


"""

import numpy as np


class FeatureParser
    def __init__(self):
        pass

    def parser(self, X):
        '''
        parse X

        for continuous features, 

        for discrete features,
        using int represent, 
        e.g. { Beijing, Shanghai, Hong Kong, Beijing }
          -> {0,1,2,0}         
        '''

        return features, X
    
    def transform(self,X):
        pass




class NbTreeClassifier
    
    def __init__(self,
                
                diffprivacy_mech = 1,
                budget = 1.0,
                
                criterion = "gini",
                random_state = None,

                max_depth = 10,
                max_candid_features = 10
                ):

        self.diffprivacy_mech = diffprivacy_mech
        self.budget = budget
        
        self.criterion = criterion
        self.random_state = random_state

        self.max_depth = max_depth
        self.max_candid_features = max_candid_features 

        # inner structure
        self._features = None
        self._tree = None

    def fit(self,
            X, y,
            sample_mask = None,
            sample_weight = None):
        
        # 1. check X, y, sample_weight
        _check_input()

        # 2. check parameter
        
        # 3. setup budget, diffprivacy

        # process features
        fparser = FeatureParser()
        self._features = FeatureParser.parser(X)

        criterion = 
        splitter  = Splitter()

        builder = TreeBuilder(self.diffprivacy_mech,
                              self.budget,
                              splitter,
                              max_depth,
                              max_candid_features)

        # 4. build tree
        builder.build(  self._tree, 
                        self._X, 
                        self._y, 
                        sample_weight, self._features)

        return self

    def predict(self, X):

