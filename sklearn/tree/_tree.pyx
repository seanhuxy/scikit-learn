# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#
# Licence: BSD 3 clause

from libc.stdlib cimport calloc, free, realloc
from libc.string cimport memcpy, memset
from libc.math cimport log as ln
from cpython cimport Py_INCREF, PyObject

from sklearn.tree._utils cimport Stack, StackRecord
from sklearn.tree._utils cimport PriorityHeap, PriorityHeapRecord

import numpy as np
cimport numpy as np
np.import_array()


cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

cdef DTYPE_t MIN_IMPURITY_SPLIT = 1e-7

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_node_samples', 'weighted_n_node_samples'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples
    ]
})


# =============================================================================
# Criterion
# =============================================================================

cdef class Criterion:
    """Interface for impurity criteria."""

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                   double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                   SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        pass

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        pass

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
           the right child to the left child."""
        pass

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of
           samples[start:pos] + the impurity of samples[pos:end]."""
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        pass

    cdef double impurity_improvement(self, double impurity) nogil:
        """Weighted impurity improvement, i.e.

           N_t / N * (impurity - N_t_L / N_t * left impurity
                               - N_t_L / N_t * right impurity),

           where N is the total number of samples, N_t is the number of samples
           in the current node, N_t_L is the number of samples in the left
           child and N_t_R is the number of samples in the right child."""
        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        # return (impurity_right + impurity_left)

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - self.weighted_n_right / self.weighted_n_node_samples * impurity_right
                          - self.weighted_n_left  / self.weighted_n_node_samples * impurity_left))

cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""
    cdef SIZE_t* n_classes
    cdef SIZE_t label_count_stride
    cdef double* label_count_left
    cdef double* label_count_right
    cdef double* label_count_total

    cdef double epsilon_per_fx

    def __cinit__(self, 
                  SIZE_t n_outputs,                         # the number of classes(outputs)
                  np.ndarray[SIZE_t, ndim=1] n_classes):    # number of distinct class value in every class
        ''' Allocate space for label_count (total,left,right)'''
        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.label_count_left = NULL
        self.label_count_right = NULL
        self.label_count_total = NULL

        # Count labels for each output
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        cdef SIZE_t k = 0
        cdef SIZE_t label_count_stride = 0

        # label_count_stride = max(number of distinct class value) 
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]
            if n_classes[k] > label_count_stride:
                label_count_stride = n_classes[k]

        self.label_count_stride = label_count_stride

        # Allocate counters
        cdef SIZE_t n_elements = n_outputs * label_count_stride
        self.label_count_left  = <double*> calloc(n_elements, sizeof(double))
        self.label_count_right = <double*> calloc(n_elements, sizeof(double))
        self.label_count_total = <double*> calloc(n_elements, sizeof(double))

        # Check for allocation errors
        if (self.label_count_left  == NULL or
            self.label_count_right == NULL or
            self.label_count_total == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""
        free(self.n_classes)
        free(self.label_count_left)
        free(self.label_count_right)
        free(self.label_count_total)

    def __reduce__(self):
        return (ClassificationCriterion,
                (self.n_outputs,
                 sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self, 
                   double    epsilon_per_fx,
                   DOUBLE_t* y, 
                   SIZE_t    y_stride,
                   DOUBLE_t* sample_weight, 
                   double    weighted_n_samples,
                   SIZE_t*   samples, 
                   SIZE_t    start, 
                   SIZE_t    end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end].

           update:
                label_count_total,(class distribution)
                weighted_n_node_samples
           """
        # Initialize fields

        self.y        = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples  = samples
        self.start    = start
        self.end      = end
        self.n_node_samples     = end - start
        self.weighted_n_samples = weighted_n_samples
        cdef double weighted_n_node_samples = 0.0

        self.noisy_n_node_samples = self.n_node_samples + laplace(1.0/epsilon)
        cdef double noisy_weighted_n_node_samples = 0.0

        # Initialize label_count_total and weighted_n_node_samples
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total

        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef SIZE_t c = 0
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        # set zero in the array of label_count_total 
        for k in range(n_outputs):
            memset(label_count_total + offset, 0, n_classes[k] * sizeof(double))
            offset += label_count_stride

        # for every sample, find its class value, increase w in corresponding index of label_count_total array
        # at the same time, update weighted_n_node_samples
        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                c = <SIZE_t> y[i * y_stride + k]
                label_count_total[k * label_count_stride + c] += w
                #label_count_total[k * label_count_stride + c] += w + laplace(1.0/epsilon)

            weighted_n_node_samples += w
            #noisy_weighted_n_node_samples += w+laplace(1.0/epsilon)

        self.weighted_n_node_samples = weighted_n_node_samples
        #self.noisy_weighted_n_node_samples = noisy_n_node_samples

        # DIFFPRIVACY
        self.epsilon_per_fx = epsilon_per_fx
        self.epsilon_4_w_n_node_samples = self.epsilon_per_fx/(1+weighted_n_node_samples)
        self.epsilon_4_update = self.epsilon_4_w_n_node_samples*weighted_n_node_samples

        self.weighted_n_node_samples += laplace(1.0/self.epsilon_4_w_n_node_samples)

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start.

        all count in label_count_left is reset to 0
        all count in label_count_right equals to the values in label_count_total
        """
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef double* label_count_left  = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef SIZE_t k = 0

        for k in range(n_outputs):
            memset(label_count_left,  0,                 n_classes[k] * sizeof(double))
            memcpy(label_count_right, label_count_total, n_classes[k] * sizeof(double))

            label_count_total += label_count_stride
            label_count_left  += label_count_stride
            label_count_right += label_count_stride

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
            the right child to the left child.

            [start:pos:end] -> [start:new_pos:end]
        """
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t y_stride = self.y_stride
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef double* label_count_left  = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t diff_w = 0.0

        # Note: We assume start <= pos < new_pos <= end

        for p in range(pos, new_pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            # DIFFPRIVACY 
            w += laplace(1.0/(epsilon_4_update*n_outputs))

            for k in range(n_outputs):
                label_index = (k * label_count_stride + <SIZE_t> y[i * y_stride + k])
                label_count_left[label_index]  += w 
                label_count_right[label_index] -= w

            diff_w += w

        self.weighted_n_left  += diff_w
        self.weighted_n_right -= diff_w

        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double epsilon, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest.

            Copy class distribution to dest
        """
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef SIZE_t k

        cdef SIZE_t i
        cdef SIZE_t zero_cnt # to count the number of zero in dest after adding laplace noise

        for k in range(n_outputs):
            memcpy(dest, label_count_total, n_classes[k] * sizeof(double))
            
            if epsilon > 0.0:
                zero_cnt = 0
                for i in range(n_classes[k]):
                    dest[i] += laplace(1.0/epsilon)
                    if dest[i] <= 0.0:
                        dest[i] = 0.0
                        zero_cnt += 1
                if zero_cnt == n_classes[k]:
                    dest[random_state.randint(0,n_classes[k])] += 1.0   # XXX random_state

            dest += label_count_stride
            label_count_total += label_count_stride

cdef class Entropy(ClassificationCriterion):
    """Cross Entropy impurity criteria.

    Let the target be a classification outcome taking values in 0, 1, ..., K-1.
    If node m represents a region Rm with Nm observations, then let

        pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = - \sum_{k=0}^{K-1} pmk log(pmk)
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total

        cdef double entropy = 0.0
        cdef double total = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            entropy = 0.0

            for c in range(n_classes[k]):
                # tmp = noisy(epsilon, label_count_total[c])
                tmp = label_count_total[c]
                if tmp > 0.0:
                    # noisy_total = noisy(epsilon, weighted_n_node_samples)
                    # entropy += noisy_total*log(tmp/noisy_total)

                    tmp /= weighted_n_node_samples
                    entropy -= tmp * log(tmp)

            total += entropy
            label_count_total += label_count_stride

        return total / n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double total_left = 0.0
        cdef double total_right = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            entropy_left = 0.0
            entropy_right = 0.0

            for c in range(n_classes[k]):
                tmp = label_count_left[c]
                if tmp > 0.0:
                    tmp /= weighted_n_left
                    entropy_left -= tmp * log(tmp)

                tmp = label_count_right[c]
                if tmp > 0.0:
                    tmp /= weighted_n_right
                    entropy_right -= tmp * log(tmp)

            total_left += entropy_left
            total_right += entropy_right
            label_count_left += label_count_stride
            label_count_right += label_count_stride

        impurity_left[0] = total_left / n_outputs
        impurity_right[0] = total_right / n_outputs

cdef class Gini(ClassificationCriterion):
    """Gini Index impurity criteria.

    Let the target be a classification outcome taking values in 0, 1, ..., K-1.
    If node m represents a region Rm with Nm observations, then let

        pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} pmk (1 - pmk)
              = 1 - \sum_{k=0}^{K-1} pmk ** 2
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total

        cdef double gini = 0.0
        cdef double total = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            gini = 0.0

            for c in range(n_classes[k]):
                # tmp = noisy(epsilon, label_count_total[c])
                tmp = label_count_total[c]
                gini += tmp * tmp

            # weighted_n_node_samples = noisy(epsilon, weighted_n_node_samples)
            # gini = gini/weighted_n_node_samples

            gini = 1.0 - gini / (weighted_n_node_samples *
                                 weighted_n_node_samples)

            total += gini
            label_count_total += label_count_stride

        return total / n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double total = 0.0
        cdef double total_left = 0.0
        cdef double total_right = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            gini_left = 0.0
            gini_right = 0.0

            for c in range(n_classes[k]):
                tmp = label_count_left[c]
                gini_left += tmp * tmp
                tmp = label_count_right[c]
                gini_right += tmp * tmp

            #gini_left = 1.0 - gini_left / (weighted_n_left *
            #                               weighted_n_left)
            #gini_right = 1.0 - gini_right / (weighted_n_right *
            #                                 weighted_n_right)

            gini_left  = gini_left / weighted_n_left                   
            gini_right = gini_right/ weighted_n_right
                                          
            total_left  += gini_left
            total_right += gini_right
            label_count_left  += label_count_stride
            label_count_right += label_count_stride

        impurity_left[0]  = total_left  / n_outputs
        impurity_right[0] = total_right / n_outputs


# =============================================================================
# Splitter
# =============================================================================

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY


cdef class Splitter:
    def __cinit__(self, 
                Criterion criterion, 
                SIZE_t max_features,
                SIZE_t min_samples_leaf, 
                object random_state):
        
        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        #self.weighted_n_samples = 0.0 
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.X = NULL
        self.X_sample_stride = 0
        self.X_fx_stride = 0
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def __dealloc__(self):
        """Destructor."""
        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self,
                   np.ndarray[DTYPE_t, ndim=2] X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight) except *:
        """Initialize the splitter.

            set samples 
                n_samples
                weighted_n_samples

            set features
                n_features
                constant_features

        """
        # Reset random state
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)

        # Initialize samples and features structures
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        
        # set samples (indices of the real X)
        j = 0
        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples


        # set features (indices of real features)
        # n_features, constant_features, feature_values
        cdef SIZE_t  n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        # Initialize X, y, sample_weight
        self.X = <DTYPE_t*> X.data
        self.X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize  # stride from one sample to the next sample
        self.X_fx_stride     = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize  # stride from one feature to the next feature, in the same sample
        
        self.y = <DOUBLE_t*> y.data
        self.y_stride = <SIZE_t> y.strides[0] / <SIZE_t> y.itemsize         # stride from the class value of one sample to that of next sample
        self.sample_weight = sample_weight

    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         double* weighted_n_node_samples) nogil:
        """Reset splitter on node samples[start:end].

            Recalculate class distribution from index 'start' to 'end'
        """
        self.start = start
        self.end   = end

        self.criterion.init(self.y,
                            self.y_stride,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find a split on node samples[start:end]."""
        pass

    cdef void node_value(self, epsilon, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest.
           Copy class distribution to dest
        """
        self.criterion.node_value(epsilon, dest)

    cdef double node_impurity(self) nogil:
        """Copy the impurity of node samples[start:end]."""
        return self.criterion.node_impurity()


cdef class BestSplitter(Splitter):
    """Splitter for finding the best split."""
    def __reduce__(self):
        return (BestSplitter, (self.criterion,
                               self.max_features,
                               self.min_samples_leaf,
                               self.random_state), self.__getstate__())

    cdef void _choose_best_split_point(self, double epsilon, DTYPE_t* Xf, SIZE_t start, SIZE_t end, SplitRecord current, SplitRecord* split) nogil:
        '''Give a feature, find the best split point'''
        
        cdef SplitRecord best
        _init_split(&best, end)

        # Reset [start,pos,end] to [start,start,end]
        self.criterion.reset()      

        p = start
        while p < end:
            # if Xf[p] and Xf[p+1] is of little difference, skip evaluating Xf[p]
            while (p + 1 < end and Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                p += 1

            # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
            #                    X[samples[p], current.feature])
            p += 1
            # (p >= end) or (X[samples[p], current.feature] >
            #                X[samples[p - 1], current.feature])

            if p < end:
                current.pos = p

                # Reject if min_samples_leaf is not guaranteed
                if (((current.pos - start) < min_samples_leaf) or
                        ((end - current.pos) < min_samples_leaf)):
                    continue

                # move sample [start,pos,end] to range[start,new_pos,end]
                self.criterion.update(current.pos)
                current.improvement = self.criterion.impurity_improvement(impurity)

                if current.improvement > feature_best.improvement:
                    self.criterion.children_impurity(&current.impurity_left,
                                                     &current.impurity_right)
                    current.threshold = (Xf[p - 1] + Xf[p]) / 2.0

                    if current.threshold == Xf[p]:
                        current.threshold = Xf[p - 1]

                    best = current  # copy

                    split[0] = best

    cdef void _exp_choose_best_split_point(self, double epsilon, DTYPE_t* Xf, SIZE_t start, SIZE_t end, SplitRecord current, SplitRecord* split) nogil:
        '''Give a feature, using exponential mechanism to find the best split point'''
        pass

    cdef void node_split(self, 
                         double epsilon,
                         double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find the best split on node samples[start:end]."""
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_fx_stride = self.X_fx_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        cdef SplitRecord feature_best

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        cdef SIZE_t n_visited_features= 0
        cdef SIZE_t n_found_constants = 0  # Number of features discovered to be constant during the split search
        cdef SIZE_t n_drawn_constants = 0  # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_known_constants = n_constant_features[0]
        cdef SIZE_t n_total_constants = n_known_constants  # n_total_constants = n_known_constants + n_found_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end

        # DIFFPRIVACY
        cdef epsilon_per_fx = epsilon / (max_features+1)

        _init_split(&best, end)
        feature_best  = <SplitRecord*> calloc(self.n_features, sizeof(SplitRecord)) # free after using
        for i in range(0, self.n_features):
            _init_split(&feature_best[i], end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.

        #  0                  n_counstant_features[0]                  n_features
        #  +                        +                                       +
        #  ------------------------------------------------------------------
        #  |    |     |     |       |       |       |       |       |       |
        #  ------------------------------------------------------------------
        #  +                        +                                       +    
        #  n_drawn_constants-->     n_known_constants,                  <-- f_i
        #                           n_total_constants -->                         
        #  
        #  n_found_coustants = |[n_known_constants,n_total_constants]|
        #

        # Stop when
        # (1) all the candidate features are constant, (f_i <= n_total_constants)
        # (2) the number of features visited >= max_features 
        #     AND 
        #     the number of features visited > the number of *constant* features visited


        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant] holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant] holds known constant features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant] holds newly found constantfeatures;
            # - [n_total_constant:f_i] holds features that haven't been drawn yet and aren't constant apriori.
            # - [f_i:n_features] holds features that have been drawn and aren't constant.


            #  0                  n_counstant_features[0]                  n_features
            #  +                        +                                       +
            #  ------------------------------------------------------------------
            #  |    |     |     |       |       |       |       |       |       |
            #  ------------------------------------------------------------------
            #       +           +       +                               +    
            #       drawn     known    total                           f_i
            #                   +-------+                            
            #                   | found |  
            #                                                                      
            #       +-------------------------------------------+-------+
            #       |   range for random select a f_j           | found | 
            #

            # Draw a feature at random
            f_j = rand_int(f_i - n_drawn_constants - n_found_constants, random_state) + n_drawn_constants

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants]
                # feautre[f_j] is a constant, move to the interval [0,drawn]
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants]
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i]

                current.feature = features[f_j]

                # Sort samples along that feature; 
                # 1. first copy the feature values for the active samples into Xf,
                #   s.t.
                #       Xf[i] == X[samples[i], j], 
                # so the sort uses the cache more effectively.
                for p in range(start, end):
                    Xf[p] = X[X_sample_stride*samples[p] + X_fx_stride*current.feature]

                sort(Xf + start, samples + start, end - start)

                # if it is constant feature
                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else: # feature f_j
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]


                    #self._choose_best_split_point(Xf, start, end, current, &feature_best[f_i])
                    self._lap_choose_best_split_point(epsilon, Xf, start, end, current, &feature_best[f_i])
                    #self._exp_choose_best_split_point(epsilon, Xf, start, end, current, &feature_best[f_i])    

                    if features_best[f_i].improvement > best.improvement
                        best = feature_best[f_i]

        # exponential mechanism
        # self._exp_choose_split_feature(epsilon, features_best, self.n_features, &best)

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride*samples[p] + X_fx_stride*best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t)*n_known_constants)

        # Copy newly found constant features to the end of `constant_features`
        memcpy(constant_features + n_known_constants,
               features          + n_known_constants,
               sizeof(SIZE_t)*n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples, SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples, SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1


# =============================================================================
# Tree builders
# =============================================================================
cdef class TreeBuilder:
    """Interface for different tree building strategies. """

    cpdef build(self, Tree tree, np.ndarray X, np.ndarray y,
                np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""
        pass


# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, 
                    Splitter splitter,

                    SIZE_t min_samples_split,   
                    SIZE_t min_samples_leaf, 
                    SIZE_t max_depth,

                    int diffprivacy,        # 0=no, 1=lap, 2=exp
                    double budget
                    ):

        self.splitter = splitter

        # criterion for splitting
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

        self.diffprivacy = diffprivacy
        self.budget      = budget

    cpdef build(self, 
                Tree       tree, 
                np.ndarray X, 
                np.ndarray y,
                np.ndarray sample_weight=None):

        """Build a decision tree from the training set (X, y)."""
        
        ##1. Check input, X, y, sample_weight -> sample_weight_ptr

        # check if dtype is correct
        if X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous: # contiguous array, 
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            if ((sample_weight.dtype != DOUBLE) or (not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE, order="C")
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity of tree
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters, which is come from __init__()
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth  = self.max_depth
        cdef SIZE_t min_samples_leaf  = self.min_samples_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split


        # DIFFPRIVACY 
        cdef double epsilon_per_action
        if self.diffprivacy > 0:
            self.epsilon_per_depth = self.budget/(self.max_depth+1)
            if self.diffprivacy == 1:
                epsilon_per_action = self.epsilon_per_depth/2.0
        else:
            self.epsilon_per_depth = -1.0
            epsilon_per_action = -1.0

        ##2. Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr)

        # a StackRecord structure
        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features

        cdef SIZE_t n_node_samples = splitter.n_samples     # number of samples on this node
        cdef double weighted_n_node_samples                 # number of weighted samples on this node
        cdef SplitRecord split
        cdef SIZE_t node_id
        cdef double threshold
        cdef bint is_leaf

        cdef SIZE_t max_depth_seen = -1 # used to record the max depth
        cdef bint first = 1             # only for root node, calculate init impurity

        cdef int rc = 0     # return value for stack.pop
        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        # push root node onto stack
        rc = stack.push(0,              # start index 
                        n_node_samples, # end index
                        0,              # depth 
                        _TREE_UNDEFINED,# parent node id
                        0,              # is left or right
                        INFINITY,       # impurity
                        0)              # number of constant features
        if rc == -1:
            # got return code -1 - out-of-memory
            raise MemoryError()

        with nogil:
            while not stack.is_empty():
                stack.pop(&stack_record)

                start   = stack_record.start
                end     = stack_record.end
                depth   = stack_record.depth
                parent  = stack_record.parent
                is_left = stack_record.is_left
                impurity= stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start

                # DIFFPRIVACY: n_node_samples = node_samples + noisy(epsilon)
                # since binary tree, the number of distinct values of all features is 2.

                noisy_n_node_samples = n_node_samples + laplace(1.0/epsilon_per_action)
                is_leaf = len(candidate_features)==0 
                           or depth >= max_depth
                           or noisy_n_node_samples/(2*n_classes) < sqrt(2)/epsilon

                #is_leaf = ((depth >= max_depth) or
                #           (n_node_samples < min_samples_split) or
                #           (n_node_samples < 2 * min_samples_leaf))

                # DIFFPRIVACY add noisy to n_node_samples, class distributions
                splitter.node_reset(start, end, &weighted_n_node_samples)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = is_leaf or (impurity <= MIN_IMPURITY_SPLIT)

                if not is_leaf:
                    # choose feature to split
                    # DIFFPRIVACY epsilon = ?
                    splitter.node_split(epsilon_per_action, impurity, &split, &n_constant_features)
                    is_leaf = is_leaf or (split.pos >= end)

                                                    # a Node structure
                node_id = tree._add_node(parent,    # parent
                                         is_left,   # left or right node
                                         is_leaf,   # leaf or inner
                                         split.feature,     # feature for split
                                         split.threshold,   
                                         impurity, 
                                         n_node_samples,    
                                         weighted_n_node_samples)

                if is_leaf:
                    # Only store class distribution for leaf node
                    
                    # DIFFPRIVCAY: noisy count, budget = epsilon_per_action 
                    splitter.node_value( epsilon_per_action ,tree.value + node_id*tree.value_stride)

                else:
                    # Push right child on stack
                    # [split.pos, end]
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break

                    # Push left child on stack
                    # [start, split.pos]
                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features)
                    if rc == -1:
                        break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()


# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The maximal depth of the tree.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            # it's small; copy for memory safety
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs).copy()

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    def __cinit__(self, 
                int                        n_features, 
                np.ndarray[SIZE_t, ndim=1] n_classes,
                int                        n_outputs):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)
        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))
        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))

    cdef void _resize(self, SIZE_t capacity) except *:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays."""
        if self._resize_c(capacity)!= 0:
            raise MemoryError()

    # XXX using (size_t)(-1) is ugly, but SIZE_MAX is not available in C89
    # (i.e., older MSVC).
    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) nogil:
        """Guts of _resize. Returns 0 for success, -1 for error."""
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        # XXX no safe_realloc here because we need to grab the GIL
        cdef void* ptr = realloc(self.nodes, capacity * sizeof(Node))
        if ptr == NULL:
            return -1
        self.nodes = <Node*> ptr
        ptr = realloc(self.value,
                      capacity * self.value_stride * sizeof(double))
        if ptr == NULL:
            return -1
        self.value = <double*> ptr

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples, double weighted_n_node_samples) nogil:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return <SIZE_t>(-1)

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        return node_id

    cpdef np.ndarray predict(self, np.ndarray[DTYPE_t, ndim=2] X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray apply(self, np.ndarray[DTYPE_t, ndim=2] X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        cdef SIZE_t n_samples = X.shape[0]
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_data = <SIZE_t*> out.data

        with nogil:
            for i in range(n_samples):
                node = self.nodes

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_data[i] = <SIZE_t>(node - self.nodes)  # node offset

        return out

    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))

        while node != end_node:
            if node.left_child != _TREE_LEAF:
                # ... and node.right_child != _TREE_LEAF:
                left = &nodes[node.left_child]
                right = &nodes[node.right_child]

                importances[node.feature] += (
                    node.weighted_n_node_samples * node.impurity -
                    left.weighted_n_node_samples * left.impurity -
                    right.weighted_n_node_samples * right.impurity)
            node += 1

        importances = importances / nodes[0].weighted_n_node_samples
        cdef double normalizer

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(np.ndarray, <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr


# =============================================================================
# Utils
# =============================================================================

# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        raise MemoryError("could not allocate (%d * %d) bytes"
                          % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience


def _realloc_test():
    # Helper for tests. Tries to allocate <size_t>(-1) / 2 * sizeof(size_t)
    # bytes, which will always overflow.
    cdef SIZE_t* p = NULL
    safe_realloc(&p, <size_t>(-1) / 2)
    if p != NULL:
        free(p)
        assert False


# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)

cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
    """Encapsulate data into a 1D numpy array of intp's."""
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> size
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data)

cdef inline SIZE_t rand_int(SIZE_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end

cdef inline double rand_double(UINT32_t* random_state) nogil:
    """Generate a random double in [0; 1)."""
    return <double> our_rand_r(random_state) / <double> RAND_R_MAX

cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)
