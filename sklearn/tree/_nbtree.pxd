
cdef struct Node:

    bint is_leaf        # If true, this is a leaf node
   
    SIZE_t  feature     # Index of the feature used for splitting this node
    DOUBLE_t threshold  # (only for continuous feature) The splitting point
    DOUBLE_t* values    # Array of class distribution 
                        #   (n_outputs, max_n_classes)
    
    SIZE_t* children    # an array, storing ids of the children of this node
    
    SIZE_t n_node_samples   # Number of samples at this node
    DOUBLE_t weighted_n_node_samples    # Weighted number of samples at this node


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
    
