"""
THis module implement no binary tree classifier
"""
import numbers
import numpy as np

from ._nbtree import Criterion
from ._nbtree import Splitter
from ._nbtree import NBTreeBuilder
from ._nbtree import Tree
from . import _nbtree

__all__ = ["NbTreeClassifier"]

# =================================================================
# Types and constants
# =================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = { "gini": _nbtree.Gini, "entropy": _nbtree.Entropy }
SPLITTERS = { "lap": _nbtree.LapSplitter,
              "exp": _nbtree.ExpSplitter }

# =================================================================
# Tree
# =================================================================

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

class NbTreeClassifier(six.with_metaclass(ABCMeta, BaseEstimator,
                                          _LearntSelectorMixin)): # XXX 

    def __init__(self,
                
                diffprivacy_mech = 1,
                budget = 1.0,
                
                criterion = "gini",
                seed = None,

                max_depth = 10,
                max_candid_features = 10
                ):

        self.diffprivacy_mech = diffprivacy_mech
        self.budget = budget
        
        self.criterion = criterion

        self.seed = seed
        if isinstance(seed, (numbers.Integral, np.integer)):
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = np.random.RandomState()

        self.max_depth = max_depth
        self.max_candid_features = max_candid_features 

        # inner structure
        self._features = None
        self._tree = None

    def _set_data(self, X, y, sample_weight):

        # XXX what it did ?
        X, = check_arrays(X, dtype=DTYPE, sparse_format="dense")

        n_samples, n_features = X.shape
      
        weighted_n_samples = 0.0
        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))
            
            for i in range(n_samples):
                weighted_n_samples += sample_weight[i]

        # y
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y, (-1,1))
       
        # class
        n_outputs = y.shape[1]

        y = np.copy(y) # XXX why copy?
        classes = []
        n_classes = []
        max_n_classes = 0
        for k in range( n_outputs ):
            classes_k, y[:,k] = np.unique(y[:,k], return_inverse=True)
            classes.append(classes_k)
            n_classes.append(classes_k.shape[0])

            if classes_k.shape[0] > max_n_classes:
                max_n_classes = classes_k.shape[0]

        n_classes = np.array( n_classes, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        
        # set data
        data = Data() 
        data.X = X
        data.y = y
        data.sample_weight = sample_weight

        data.n_samples = n_samples
        data.weighted_n_samples = weighted_n_samples

        data.n_features = n_features
        data.features = FeatureParser.parser(X) # XXX array of Feature

        # max_n_feature_values
        for i in range(n_features):
            if features[i].n_values > max_n_feature_values:
                max_n_feature_values = features[i].n_values

        # classes
        data.n_outputs = n_outputs
        data.classes   = classes
        data.n_classes = n_classes
        data.max_n_classes = max_n_classes
    
        return data

    def fit(self,
            X, y,
            sample_weight = None):
       
        # random_state
        random_state = self.random_state

        # 1. set data
        data = _set_data(X, y, sample_weight)
        self.data = data

        # 2. check parameter
        # XXX max depth
        max_depth = self.max_depth 
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero.")
        
        # 3. setup budget, diffprivacy
        diffprivacy = self.diffprivacy
        budget = self.budget
       
        criterion = CRITERIA_CLF[self.criterion](self.data)

        splitter = SPLITTERS[ diffprivacy ]
                (criterion, max_candid_features, random_state)  

        tree = Tree(data.n_features, data.n_classes, data.n_outputs)
        self.tree_ = tree

        builder = TreeBuilder(diffprivacy_mech,
                              budget,
                              splitter,
                              max_depth,
                              max_candid_features)

        # 4. build tree
        builder.build( tree, data )
        
        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def predict(self, X):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        if getattr(X, "dtype", None) != DTYPE or X.ndim != 2:
            X = array2d(X, dtype=DTYPE)

        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise Exception("Tree not initialized. Perform a fit first")

        if self.data.n_features != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.data.n_features, n_features))

        proba = self.tree_.predict(X)

        # Classification
        if isinstance(self, ClassifierMixin):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)

            else:
                predictions = np.zeros((n_samples, self.n_outputs_))

                for k in xrange(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1),
                        axis=0)

                return predictions
