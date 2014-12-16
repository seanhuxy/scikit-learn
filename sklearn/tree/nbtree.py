"""
THis module implement no binary tree classifier
"""
from __future__ import division

import numbers
import numpy as np
from abc import ABCMeta, abstractmethod
from warnings import warn

from ..base import BaseEstimator, ClassifierMixin
from ..externals import six
from ..externals.six.moves import xrange
from ..feature_selection.from_model import _LearntSelectorMixin
from ..utils import array2d
from ..utils.validation import check_arrays

from ._nbtree import DataObject
# from ._nbtree import FeatureParser
from ._nbtree import Criterion
from ._nbtree import Splitter
from ._nbtree import LapSplitter
from ._nbtree import ExpSplitter
from ._nbtree import NBTreeBuilder
from ._nbtree import Tree
from . import _nbtree

# __all__ = ["NBTreeClassifier"]

# =================================================================
# Types and constants
# =================================================================

DTYPE  = _nbtree.DTYPE     # XXX
DOUBLE = _nbtree.DOUBLE

NO_DIFF_PRIVACY_MECH  = 0
LAP_DIFF_PRIVACY_MECH = 1
EXP_DIFF_PRIVACY_MECH = 2


CRITERIA_CLF = { "gini": Criterion, "entropy": Criterion } # XXX
SPLITTERS = { NO_DIFF_PRIVACY_MECH  : LapSplitter, 
              LAP_DIFF_PRIVACY_MECH : LapSplitter,
              EXP_DIFF_PRIVACY_MECH : ExpSplitter }

# =================================================================
# Tree
# =================================================================
class NBTreeClassifier(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin,
                                          _LearntSelectorMixin)): # XXX 

    def __init__(self,
                
                diffprivacy_mech = NO_DIFF_PRIVACY_MECH ,
                budget = -1.0,
                
                criterion = "gini",
                seed = 1,

                max_depth = 5,
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
        self._tree = None

    def fit(self,
            X, y,
            meta,
            sample_weight = None,
            debug = False
            ):
       
        # random_state
        random_state = self.random_state

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
       
        X, = check_arrays(X, dtype=DTYPE, sparse_format="dense")
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

       
        # 1. init Data
        dataobject = DataObject(X, y, meta, sample_weight)

        # 2. check parameter
        # XXX max depth
        max_depth = self.max_depth 
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero.")
        
        # 3. setup budget, diffprivacy
        diffprivacy_mech = self.diffprivacy_mech
        budget = self.budget
        max_candid_features = self.max_candid_features 

        criterion = CRITERIA_CLF[self.criterion](dataobject, random_state)

        splitter = SPLITTERS[ diffprivacy_mech ](criterion, max_candid_features, random_state)  

        tree = Tree(dataobject)
        self._tree = tree

        builder = NBTreeBuilder(diffprivacy_mech,
                              budget,
                              splitter,
                              max_depth,
                              max_candid_features,
                              random_state)

        # 4. build tree
        builder.build( tree, dataobject, debug)
        self.data = dataobject 
        if self.data.n_outputs == 1:
            self.data.n_classes = self.data.n_classes[0]
            self.data.classes = self.data.classes[0]
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

        if self._tree is None:
            raise Exception("Tree not initialized. Perform a fit first")

        if self.data.n_features != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.data.n_features, n_features))

        proba = self._tree.predict(X)

        # Classification
        if isinstance(self, ClassifierMixin):
            if self.data.n_outputs == 1:
                return self.data.classes.take(np.argmax(proba, axis=1), axis=0)

            else:
                predictions = np.zeros((n_samples, self.data.n_outputs))

                for k in xrange(self.data.n_outputs):
                    predictions[:, k] = self.data.classes[k].take(
                        np.argmax(proba[:, k], axis=1),
                        axis=0)

                return predictions
