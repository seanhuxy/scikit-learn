"""
THis module implement no binary tree classifier
"""
from __future__ import division

import numbers
import numpy as np
from abc import ABCMeta, abstractmethod
from warnings import warn

from six import string_types

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

CRITERIA_CLF = { "gini": _nbtree.Gini, "entropy": _nbtree.Entropy, "lapentropy": _nbtree.LapEntropy } # XXX
SPLITTERS = { NO_DIFF_PRIVACY_MECH  : ExpSplitter, 
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
                max_candid_features = 10,
                min_samples_leaf = 0,

                print_tree = True ,
                is_prune = True,
                CF = 0.25,
                ):

        if isinstance(diffprivacy_mech, string_types):
            if diffprivacy_mech is "no":
                diffprivacy_mech = NO_DIFF_PRIVACY_MECH
            elif diffprivacy_mech in ["laplace", "lap", "l"]:
                diffprivacy_mech = LAP_DIFF_PRIVACY_MECH
            elif diffprivacy_mech in ["exponential", "exp", "e"]:
                diffprivacy_mech = EXP_DIFF_PRIVACY_MECH
            else:
                raise ValueError("diffprivacy_mech %s is illegal"%diffprivacy_mech)
        elif isinstance(diffprivacy_mech, (numbers.Integral, np.integer)):
            if diffprivacy_mech not in [NO_DIFF_PRIVACY_MECH, LAP_DIFF_PRIVACY_MECH, EXP_DIFF_PRIVACY_MECH]:
                raise ValueError
        else:
            raise ValueError

        self.diffprivacy_mech = diffprivacy_mech
        self.budget = budget
       
        if diffprivacy_mech == LAP_DIFF_PRIVACY_MECH:
            criterion = "lapentropy"
        self.criterion = criterion
        
        self.seed = seed
        if isinstance(seed, (numbers.Integral, np.integer)):
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = np.random.RandomState()

        self.max_depth = max_depth
        self.max_candid_features = max_candid_features 
        self.min_samples_leaf = min_samples_leaf

        self.print_tree = print_tree
        self.is_prune = is_prune
        self.CF = CF

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

        criterion = CRITERIA_CLF[self.criterion](dataobject, random_state, debug)

        splitter = SPLITTERS[ diffprivacy_mech ](criterion, max_candid_features, random_state, debug)

        tree = Tree(dataobject, debug)
        self._tree = tree


        print "# ====================================="
        print "# Begin to build tree"
        print "# b={0}, d={1}, prune={2}, CF={3}".format(budget, max_depth, self.is_prune, self.CF)
        builder = NBTreeBuilder(diffprivacy_mech,
                              budget,
                              splitter,
                              max_depth,
                              max_candid_features,
                              self.min_samples_leaf,
                              random_state,
                              self.print_tree,
                              self.is_prune,
                              self.CF)


        # print "Begin to build the tree"
        # 4. build tree
        builder.build( tree, dataobject, debug)
        # print "Finished to build the tree"
        
        print "# ====================================="

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
        debug = False
        if debug:
            print "get into predict"
        
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

        if debug:
            print "get out of tree.predict"

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

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        if getattr(X, "dtype", None) != DTYPE or X.ndim != 2:
            X = array2d(X, dtype=DTYPE)

        n_samples, n_features = X.shape

        if self._tree is None:
            raise Exception("Tree not initialized. Perform a fit first.")

        if self.data.n_features != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.data.n_features, n_features))

        proba = self._tree.predict(X)

        if self.data.n_outputs == 1:
            proba = proba[:, :self.data.n_classes]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in xrange(self.data.n_outputs_):
                proba_k = proba[:, k, :self.data.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba

