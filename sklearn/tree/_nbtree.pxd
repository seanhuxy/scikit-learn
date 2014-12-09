import numpy as np
cimport numpy as np

from numpy.random import RandomState

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# ====================================================================
# Data
# ===================================================================
cdef struct Feature:

    cdef int type       # continuous or discrete feature
    
    cdef int n_values   # the number of distinct value of the feature
                        # for continuous feature, = 2  

    cdef double max     # for continuous feature only
    cdef double min     # for continuous feature only

# finished,  TODO:
#   check type 
cdef class Data:

    DOUBLE_t* X
    DOUBLE_t* y
    DOUBLE_t* sample_weight
    
    SIZE_t  X_sample_stride  
    SIZE_t  X_feature_stride 
    SIZE_t  y_stride 

    SIZE_t  n_samples 
    SIZE_t  weighted_n_samples 

    Feature* features
    SIZE_t  n_features
    SIZE_t  max_n_feature_values # max(number of distint values of all feature)
    SIZE_t  n_continuous_features

    SIZE_t    n_outputs
    SIZE_t*   classes
    np.ndarray[SIZE_t, ndim=1] n_classes 
    SIZE_t    max_n_classes    # max(number of distinct class label)
# =======================================================================
# Criterion
# =======================================================================
cdef class Criterion:

    cdef Data   data

    cdef SIZE_t* samples_win            # reserved, from splitter
    cdef SIZE_t n_node_samples          # reserved 
    cdef SIZE_t weighted_n_node_samples
    
    cdef SIZE_t start
    cdef SIZE_t end
    
    cdef DOUBLE_t* label_count_total    # shape[n_outputs][max_n_classes]
    cdef DOUBLE_t* label_count          # shape[max_n_feature_values][n_outputs][max_n_classes] 

    cdef SIZE_t label_count_stride      # = max_n_classes
    cdef SIZE_t feature_stride          # = n_outputs * max_n_classes

    # Methods
    cdef void init(self,
                   Data data,
                   SIZE_t* samples_win,
                   SIZE_t start,
                   SIZE_t end) nogil
    cdef void update(self, 
                     SIZE_t* samples_win,
                     SplitRecord split_record,
                     double* Xf) nogil
    cdef double node_impurity(self, DOUBLE_t* label_count, SIZE_t wn_samples) nogil
    cdef void node_value(self, double* dest) nogil

    cdef double improvement(self, SIZE_t* wn_subnodes_samples, SIZE_t n_subnodes) nogil


# =======================================================================
# Splitter
# =======================================================================
cdef struct SplitRecord:
    SIZE_t feature_index    # the index of the feature to split
    SIZE_t threshold        # for continuous feature only
    DOUBLE_t improvement    # the improvement by selecting this feature
   
    SIZE_t  n_subnodes
    SIZE_t* n_subnodes_samples
    SIZE_t* wn_subnodes_samples 

cdef class Splitter:

    cdef public Criterion criterion
    cdef Data data
    cdef RandomState random_state

    cdef SIZE_t* samples_win
    cdef SIZE_t  start, end

    cdef SIZE_t* features_win
    cdef SIZE_t  n_features
    
    cdef DOUBLE_t* feature_values        # .temp

    cdef SIZE_t max_candid_features      # reserved

    # Methods
    cdef void init(self, Data data)

    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         double* weighted_n_node_samples) nogil

    cdef void _choose_split_point(self) # reserved
    cdef void _choose_split_feature(self, 
                                SplitRecord* best, SplitRecord* records, 
                                int size, double epsilon)

    cdef void node_split(self,
                         SplitRecord* split_record,
                         SIZE_t* n_node_features, 
                         double epsilon) nogil

    cdef void node_value(self, double* dest) nogil
# ==================================================================
# Builder
# ==================================================================
cdef class NBTreeBuilder:

    # diffprivacy parameter
    SIZE_t diffprivacy_mech
    double budget

    # tree parameter
    SIZE_t max_depth
    SIZE_t max_candid_features
    
    # inner structure
    Splitter    splitter
    Data        data    # X, y and metadata
    Tree        tree
    RandomState random_state

    cpdef build(self, Tree tree, Data data) 

# ==================================================================
# Tree structure
# =================================================================
cdef struct Node:
    bint is_leaf        # If true, this is a leaf node

    # For inner node
    SIZE_t  feature     # Index of the feature used for splitting this node
    DOUBLE_t threshold  # (only for continuous feature) The splitting point
    
    SIZE_t* children    # an array, storing ids of the children of this node
    SIZE_t  n_children  # size = feature.n_values
    
    # For leaf node
    DOUBLE_t* values    # (only for leaf node) Array of class distribution 
                        #   (n_outputs, max_n_classes)
    SIZE_t  label       # class label, max_index(values)

    # reserved
    SIZE_t n_node_samples   # Number of samples at this node
    DOUBLE_t weighted_n_node_samples   # Weighted number of samples at this node

cdef class Tree:

    # Input/Output layout
    cdef public SIZE_t n_features
    cdef SIZE_t* n_classes          # Number of diff labels of each class in y
    cdef public* SIZE_t n_outputs   # Number of outputs in y
    cdef public SIZE_t max_n_classes# max(n_classes)

    # Inner structures

    cdef public SIZE_t max_depth    # Max depth of the tree
    cdef public SIZE_t capacity     # Capacity of trees

    cdef Node* nodes                # Array of nodes
    
    cdef SIZE_t _add_node(self, 
                SIZE_t parent,
                bint is_leaf,
                ) nogil

    cpdef np.ndarray predict(self, np.ndarray(DTYPE_t, ndim=2] X)
    
