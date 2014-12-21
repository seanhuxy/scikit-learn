__author__ = 'deng'

from array import array
from collections import Mapping
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

from utils import six

FEATURE_CONTINUOUS = 0
FEATURE_DISCRET = 1

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
        self.interval = np.NaN

        if False:
            self.max_ = np.NaN 
            self.min_ = np.NaN
            self.interval = np.NaN

    def discretize(self, v):
         
        try:
            index = int( (v - self.min_)/ self.interval )
        except ValueError:
            print v, self.min_, self.max_ 
        if index >= self.n_values:
            if index == self.n_values and v == self.max_:
                index = self.n_values-1
            else:
                error = "Discretizing continous value %f error, min %f, max %f, bins %u"%(v, self.min_, self.max_, self.n_values)
                raise ValueError(error)

        return index

    def __str__(self):
        if self.type == FEATURE_DISCRET:
            ret = 'name {0}\tn_values {1}\nindices {2}\nvalues {3}'.format(self.name, self.n_values, self.indices, self.values)
        else:
            ret = 'name {0}\tn_values {1}\tmax_ {2}\tmin_ {3}\tinterval {4}'.format(self.name, self.n_values, self.max_, self.min_, self.interval)
        
        return ret 

class Dict_Vectorizer():
    """
        use fit_transform to transform the raw dataset to matrix.
    """
    def __init__(self):
        pass 
    def feature_name(self, f):
        return self.features_[f].name

    def feature_value(self, f, v):
        feature = self.features_[f]
        if   feature.type == FEATURE_CONTINUOUS and not self.discretize:
            return v
        elif feature.type == FEATURE_CONTINUOUS and self.discretize:
            return feature.discretize(v) 
        else:
            return self.features_[f].values[v]
   
    def print_features(self):
        for i in range(len(self.features_)):
            print "feature[{0}]:".format(i)
            print self.features_[i]
            print "\n"

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
                        feature.n_values = 0
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

    def _transform(self, X, y, discretize):
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

            if discretize and feature.type == FEATURE_CONTINUOUS:
                n_bins = 10 if (y is None or y[feature.name] is None) else y[feature.name]
                feature.n_values = n_bins    
                feature.interval = (feature.max_ - feature.min_)/n_bins
                if feature.interval <= 0.0:
                    feature.interval = 0.1
            if not discretize and feature.type == FEATURE_CONTINUOUS:
                feature.n_values = 2
                

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
                    elif discretize and feature.type == FEATURE_CONTINUOUS:
                        X_row[index] = feature.discretize(v)
                    elif not discretize and feature.type == FEATURE_CONTINUOUS:
                        X_row[index] = v
                    else:
                        raise ValueError

            X_result.append(X_row)
            y_result.append(y_row)

        # debug
        if False:
            print 'n_features =', n_features
            print 'feature dict:\n', feature_name2index 
            print 'features:\n'
            for f in featFalse:#
                print f
            print 'class\n', class_feature
        
        return np.array(X_result), np.array(y_result), self


    def fit_reverse(self, X, y):
        
        n_samples, n_features = X.shape
        n_classes = y.shape[1]


    def fit_transform(self, X, bins=10, discretize=True):
        ''' return X, y, metadata'''
        self.discretize = discretize
        self.fit(X)
        return self._transform(X, bins, discretize)
