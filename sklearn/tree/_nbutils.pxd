# ==================================
# Utils
# =================================
import numpy as np
cimport numpy as np

ctypedef np.npy_intp SIZE_t 

cdef struct StackRecord:
    SIZE_t start
    SIZE_t end
    SIZE_t depth
    SIZE_t parent
    SIZE_t index        # the index of this node in parent's children array
    SIZE_t n_node_features

cdef class Stack:
    cdef SIZE_t capacity
    cdef SIZE_t top
    cdef StackRecord* stack_

    cdef bint is_empty(self) nogil

    cdef int _resize(self) nogil
    #cdef int push(self, StackRecord sr) nogil
    cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t depth,
            SIZE_t parent, SIZE_t index, SIZE_t n_node_features) nogil
    cdef int pop(self, StackRecord* sr) nogil


