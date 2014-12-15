__author__ = 'deng'

from array import array
from collections import Mapping
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from utils import six

FEATURE_DISCRET = 0
FEATURE_CONTINUOUS = 1

class Feature():

    def __init__(self):
        self.type = None
        self.name = None    # string 'city'
        
        self.n_values = 0
        
        # for discret feature
        self.indices = {}   # dict  { 'beijing' : 0, 'shanghai': 1 } 
        self.values  = []   # list  [ 'beijing', 'shanghai' ]
       
        # for continuous feature
        self.max_ = None 
        self.min_ = None
        self.interval = -1

        if False:
            self.max_ = np.NaN 
            self.min_ = np.NaN
            self.interval = np.NaN

    def __str__(self):
        ret = 'a instance of Feature, \n'
        if self.type is None:
            ret += "Invalid feature type"
        elif self.type == FEATURE_DISCRET:
            ret += ' name is %s,\n n_values is %d,\n indices is %s, \n values is %s' %( self.name, self.n_values, self.indices, self.values )
        else:
            ret += ' name is %s,\n n_values is %d,\n max_ is %f, min_ is %f, interval is %f' %( self.name, self.n_values, self.max_, self.min_, self.interval )
        
        return ret 

class Dict_Vectorizer():
    """
        use fit_transform to transform the raw dataset to matrix.
    """
    def __init__(self, sort=True):
        self.sort = sort

    def fit(self, X):
        """ init self.feature_names_ and self.feature_values_
        :param X: raw matrix of the dataset
        :return: self
        """
        n_features = len(X[0])

        features = []        # list of Feature()
        class_feature = None # Feature()
        
        feature_name2index = {}

        for record in X:
            for f, v in six.iteritems(record):
           
                feature = None 
                if f == 'class':
                    if class_feature is None:
                        class_feature = Feature()
                        class_feature.type = FEATURE_DISCRET      # class must be discret
                        class_feature.name = f
                        feature_name2index[f] = n_features-1
                    feature = class_feature
                elif f not in feature_name2index.keys():
                    feature_name2index[f] = len(features)
                    feature = Feature()
                    feature.name = f
                    features.append(feature)
                    if len(features) >= n_features:
                        raise ValueError(" len of features greater than n_features ")
                else:
                    feature = features[ feature_name2index[f] ]

                # if string
                if isinstance(v, six.string_types):
                    if feature.type is None:
                        feature.type = FEATURE_DISCRET
                    if feature.type == FEATURE_CONTINUOUS:
                        raise ValueError("feature type error")
                    
                    if v not in feature.values:
                        index = len(feature.values)
                        feature.indices[v] = index
                        feature.values.append(v)
                        
                        feature.n_values += 1

                else:
                    if feature.type is None:
                        feature.type = FEATURE_CONTINUOUS
                    if feature.type == FEATURE_DISCRET:
                        raise ValueError("feature type error")                    
                    
                    if feature.max_ is None:
                        feature.max_ = v
                    if feature.min_ is None:
                        feature.min_ = v

                    feature.max_ = feature.max_ if feature.max_ > v else v
                    feature.min_ = feature.min_ if feature.min_ < v else v
      
        
        self.n_features_ = n_features        
        self.features_ = features
        self.class_feature_ = class_feature
        self.feature_name2index_ = feature_name2index 
        
        # debug
        if False:
            print 'n_features =', n_features
            print 'feature dict:\n', feature_name2index 
            print 'features:\n'
            for f in features:
                print f
            print 'class\n', class_feature
        
        return self

    def _transform(self, X, y):
        """
        :param X: raw matrix of the train
        :param y: list of param which is used to divide the numerical features
        :return: new matrix of the data set
        """
        features = self.features_
        class_feature = self.class_feature_
        n_features = self.n_features_
       
        feature_name2index = self.feature_name2index_
        
        for feature in features:
            n_bins = 10 if (y is None or y[feature.name] is None) else y[feature.name]
            if feature.type == FEATURE_CONTINUOUS:
                feature.n_values = n_bins    
                feature.interval = (feature.max_ - feature.min_)/n_bins

        X_result = []
        y_result = []
        for record in X:
            X_row = np.zeros( n_features - 1 )
            y_row = np.zeros( 1 )

            for f, v in six.iteritems(record):
               
                if f == 'class':
                    y_row[0] = class_feature.indices[v]
                else:
                    index = feature_name2index[f]
                    feature = features[index]
                    if feature.type == FEATURE_DISCRET:
                        X_row[index] = feature.indices[v]
                    else:
                        X_row[index] = int( (v - feature.min_)/ feature.interval )

            X_result.append(X_row)
            y_result.append(y_row)

        # debug
        if True:
            print 'n_features =', n_features
            print 'feature dict:\n', feature_name2index 
            print 'features:\n'
            for f in features:
                print f
            print 'class\n', class_feature
        
        return np.array(X_result), np.array(y_result), self

    def feature_name(self, feature_index):
        pass

    def feature_value(self, feature_index, value_index):
        pass

    def class_label(self, class_label_index):
        pass

    def origin_X(self, X_index):
        pass


    def get_features(self):
        pass

    def fit_transform(self, X, bins=10, discretize=True):
        ''' return X, y, metadata'''
        self.fit(X)
        return self._transform(X, bins)
