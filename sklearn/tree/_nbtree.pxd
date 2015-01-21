import numpy as np
cimport numpy as np

from numpy.random import RandomState
#cimport numpy.random.RandomState

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# ====================================================================
# Data
# ===================================================================
cdef struct Feature:
    char* name
    SIZE_t type     # continuous or discrete feature
    
    SIZE_t n_values # the number of distinct value of the feature
                    # for continuous feature, = 2  

    #DOUBLE_t max    # for continuous feature only
    #DOUBLE_t min    # for continuous feature only

cdef struct Data:
    DTYPE_t* X
    DOUBLE_t* y
    DOUBLE_t* sample_weight
    
    SIZE_t  X_sample_stride  
    SIZE_t  X_feature_stride 
    SIZE_t  y_stride 
    
    SIZE_t  n_samples 
    DOUBLE_t  weighted_n_samples 
    
    Feature* features
    SIZE_t  n_features
    SIZE_t  max_n_feature_values # max(number of distint values of all feature)
    SIZE_t  n_continuous_features
    SIZE_t  avg_n_feature_values

    SIZE_t  n_outputs
    # cdef      SIZE_t* classes
    SIZE_t* n_classes
    SIZE_t  max_n_classes    # max(number of distinct class label)

# wrapper class for Data
cdef class DataObject:

    cdef Data* data

    cdef public SIZE_t n_features
    cdef public SIZE_t n_outputs
    cdef public object classes
    cdef public object n_classes

# =======================================================================
# Criterion
# =======================================================================
cdef class Criterion:

    cdef Data*   data

    cdef SIZE_t* samples_win            # reserved, from splitter
    cdef SIZE_t n_node_samples          # reserved 
    cdef DOUBLE_t weighted_n_node_samples
    
    cdef SIZE_t start
    cdef SIZE_t end
    cdef SIZE_t pos                     # for continuous feature
    
    cdef double* label_count_total    # shape[n_outputs][max_n_classes]
    cdef double* label_count          # shape[max_n_feature_values][n_outputs][max_n_classes] 

    cdef SIZE_t label_count_stride      # = max_n_classes
    cdef SIZE_t feature_stride          # = n_outputs * max_n_classes

    cdef DOUBLE_t sensitivity 

    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    cdef bint debug

    # Methods
    cdef void init(self,
                   Data* data,
                   SIZE_t* samples_win,
                   SIZE_t start,
                   SIZE_t end) # nogil

    cdef void reset(self, SIZE_t feature_index)   # only used for continuous feature
    # for continuous feature
    cdef void cupdate(self, SIZE_t* samples_win, SIZE_t feature_index, SIZE_t new_pos) 
    cdef void dupdate(self,                                      # for discret feature 
                     SIZE_t* samples_win,
                     SIZE_t  feature_index,
                     DTYPE_t* Xf) # nogil


    cdef DOUBLE_t node_impurity(self) # nogil
    cdef DOUBLE_t children_impurity(self, double* label_count, DOUBLE_t wn_samples, DOUBLE_t epsilon) # nogil

    cdef DOUBLE_t improvement(self, DOUBLE_t* wn_subnodes_samples, SIZE_t n_subnodes, 
                            DOUBLE_t impurity, DOUBLE_t epsilon) # nogil

    cdef void node_value(self, DOUBLE_t* dest) # nogil

    cdef void print_distribution(self, DOUBLE_t* dest) #nogil

# =======================================================================
# Splitter
# =======================================================================
cdef struct SplitRecord:
    SIZE_t feature_index    # the index of the feature to split
    DOUBLE_t improvement    # the improvement by selecting this feature
    DOUBLE_t threshold      # for continuous feature only
    SIZE_t pos              # for continuous feature

    #SIZE_t  n_subnodes
    #SIZE_t* n_subnodes_samples
    #DOUBLE_t* wn_subnodes_samples 

cdef class Splitter:

    cdef int debug

    cdef public Criterion criterion
    cdef Data* data

    cdef SIZE_t* samples_win
    cdef SIZE_t  start, end

    cdef SIZE_t* features_win
    cdef SIZE_t  n_features
    
    cdef DTYPE_t* feature_values    # temp

    cdef SplitRecord* records       # temp
    cdef SIZE_t*  positions         # temp
    cdef double*  improvements
    cdef double*  improvements_exp  # only for exponential mech
    cdef double*  weights           # temp

    cdef SIZE_t*   n_sub_samples
    cdef DOUBLE_t* wn_sub_samples


    cdef SIZE_t max_candid_features # reserved

    cdef object random_state        # Random state
    cdef UINT32_t rand_r_state      # sklearn_rand_r random number state

    # Methods
    cdef void init(self, Data* data, SIZE_t max_candid_features)

    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         DOUBLE_t* weighted_n_node_samples) # nogil

    cdef void _choose_split_point(self, SplitRecord* best, DTYPE_t* Xf, 
                                    DOUBLE_t impurity, DOUBLE_t epsilon) # nogil
    cdef SIZE_t _choose_split_feature(self, 
                                SplitRecord* records, 
                                SIZE_t size, DOUBLE_t epsilon) # nogil

    cdef void node_split(self,
                         SplitRecord* split_record,
                         SIZE_t* n_node_features, 
                         DOUBLE_t impurity,
                         SIZE_t  diffprivacy_mech,
                         DOUBLE_t epsilon) except *

    cdef SIZE_t node_max_n_feature_values(self, SIZE_t n_node_samples)

    cdef void node_value(self, DOUBLE_t* dest) # nogil

    cdef DOUBLE_t node_impurity(self) # nogil

# ==================================================================
# Builder
# ==================================================================
cdef class NBTreeBuilder:

    # diffprivacy parameter
    cdef SIZE_t diffprivacy_mech
    cdef DOUBLE_t budget

    # tree parameter
    cdef SIZE_t max_depth
    cdef SIZE_t max_candid_features
    cdef SIZE_t min_samples_leaf 

    # inner structure
    cdef Splitter splitter
    cdef Data*  data     # X, y and metadata
    cdef Tree   tree

    cdef object random_state    # Random state
    cdef UINT32_t rand_r_state  # sklearn_rand_r random number state
    
    cdef bint print_tree        # if True, print the tree to stdout 
    cdef bint is_prune
    cdef double CF

    cpdef build(self, Tree tree, DataObject dataobject, bint debug)

# ==================================================================
# Tree structure
# =================================================================
cdef struct Node:
    
    SIZE_t parent
    bint is_leaf        # If true, this is a leaf node

    # For inner node
    SIZE_t  feature     # Index of the feature used for splitting this node
    DOUBLE_t threshold  # (only for continuous feature) The splitting point
    
    DOUBLE_t impurity   # reserved. Impurity of the node (i.e., the value of the criterion)
    DOUBLE_t improvement
    
    SIZE_t* children    # an array, storing ids of the children of this node
    SIZE_t  n_children  # size = feature.n_values

    # reserved
    SIZE_t n_node_samples   # Number of samples at this node
    DOUBLE_t weighted_n_node_samples   # Weighted number of samples at this node
    DOUBLE_t noise_n_node_samples

cdef class Tree:

    # Input/Output layout
    cdef Feature* features          # XXX
    cdef Data* data

    cdef public SIZE_t n_features
    cdef SIZE_t* n_classes          # Number of diff labels of each class in y
    cdef public SIZE_t n_outputs    # Number of outputs in y
    cdef public SIZE_t max_n_classes# max(n_classes)

    # Inner structures
    cdef public SIZE_t capacity     # Capacity of trees
    cdef public SIZE_t node_count   # Counter for node IDs
    cdef Node*   nodes              # Array of nodes
    cdef double* value              # Array of class distribution, (capacity, n_outputs, max_n_classes)
    cdef SIZE_t value_stride        # = n_outputs*max_n_classes

    cdef public SIZE_t max_depth    # Max depth of the tree

    cdef void _resize(self, SIZE_t capacity)
    cdef int _resize_c(self, SIZE_t capacity=*) # nogil

    cdef SIZE_t _add_node(self, 
                SIZE_t parent,
                SIZE_t index,
                bint is_leaf,
                SIZE_t feature,
                DOUBLE_t threshold,
                SIZE_t n_children,
                DOUBLE_t importance,
                SIZE_t n_node_samples,
                DOUBLE_t weighted_n_node_samples,
                DOUBLE_t noise_n_node_samples
                ) # nogil

    cpdef np.ndarray predict(self, np.ndarray[DTYPE_t, ndim=2] X)
    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)
    cpdef np.ndarray apply(self, np.ndarray[DTYPE_t, ndim=2] X)

    cdef void calibrate_n_node_samples(self, SIZE_t node_id, DOUBLE_t fixed_n_node_samples)
    cdef void calibrate_class_distribution(self, SIZE_t node_id) 
    cdef double n_errors(self, double* counts, double noise_n_node_samples)
    cdef double leaf_error(self, SIZE_t node_id, double CF)
    cdef double node_error(slef, SIZE_t node_id, double CF)
    cdef void prune(self, SIZE_t node_id, double CF)

    cdef void compute_node_feature_importance(self, SIZE_t node_id, np.ndarray importances)
    cpdef np.ndarray compute_feature_importances(self, normalize=*)

    cdef void print_tree(self)
    cdef void print_node(self, SIZE_t node_id, Node* parent, SIZE_t feature, SIZE_t index, SIZE_t depth)

