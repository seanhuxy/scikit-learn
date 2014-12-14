from libc.stdio cimport printf
from libc.stdlib cimport calloc, free, realloc
from libc.string cimport memcpy, memset
from libc.math   cimport log, exp, sqrt
from cpython cimport Py_INCREF, PyObject

from sklearn.tree._nbutils cimport Stack, StackRecord

from sklearn.tree._tree cimport sort, rand_int, rand_double # XXX

import numpy as np
cimport numpy as np
np.import_array() # XXX

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

cdef double INFINITY = np.inf

cdef enum: 
    _TREE_UNDEFINED = -1

cdef SIZE_t INITIAL_STACK_SIZE = 10

cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': [  'parent', 'is_leaf',      # XXX 
                'feature', 'threshold', 'impurity',
                'children', 'n_children', # XXX
                'n_node_samples', 'weighted_n_node_samples'],
    'formats': [np.intp, np.bool_, 
                np.intp, np.float64, np.float64,
                np.intp, np.intp,  
                np.intp, np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).parent,
        <Py_ssize_t> &(<Node*> NULL).is_leaf,   # XXX
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).children,  # XXX
        <Py_ssize_t> &(<Node*> NULL).n_children,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples
    ]
})


cpdef public SIZE_t NO_DIFF_PRIVACY_MECH  = 0
cpdef public SIZE_t LAP_DIFF_PRIVACY_MECH = 1
cpdef public SIZE_t EXP_DIFF_RPIVACY_MECH = 2

cpdef public SIZE_t NO_DIFF_PRIVACY_BUDGET = -1

cpdef SIZE_t NO_THRESHOLD = -1
cpdef SIZE_t NO_FEATURE = -1
cpdef SIZE_t FEATURE_CONTINUOUS = 0
cpdef SIZE_t FEATURE_DISCRETE   = 1

# ====================================================================
# Criterion
# ====================================================================

cdef class Criterion:

    def __cinit__(self, DataObject dataobject, object random_state):
        '''
            allocate:
                label_count,
                label_count_total

            set:
                label_count_stride, = max_n_classes 
                feature_stride,     = max_n_classes * n_outputs 
        '''
        self.random_state = random_state 
        self.rand_r_state = random_state.randint(0, RAND_R_MAX)
       
        cdef Data data = dataobject.data

        # self.samples_win = NULL

        self.start = 0
        self.end   = 0

        self.label_count_stride = data.max_n_classes 

        cdef SIZE_t feature_stride = data.n_outputs * self.label_count_stride
        
        self.label_count_total = <DOUBLE_t*> calloc(feature_stride, sizeof(DOUBLE_t))
        self.label_count = <DOUBLE_t*> calloc(feature_stride*data.max_n_feature_values, sizeof(DOUBLE_t))
        self.feature_stride = feature_stride

        if self.label_count_total == NULL or self.label_count == NULL:
            raise MemoryError()

        self.sensitivity = 2.0   # gini
    
        self.data = data

    cdef void init(self, 
                   Data      data,
                   SIZE_t*   samples_win, 
                   SIZE_t    start, 
                   SIZE_t    end,
                   ) nogil:
        '''
        For one node, called once,
        update class distribution in this node
            
        fill: 
            label_count_total, shape[n_outputs][max_n_classes]    
        update:
            weighted_n_node_samples  
        '''
        
        # Initialize fields
        self.start    = start
        self.end      = end
        self.n_node_samples = end - start
      
        # XXX shallow copy Data structure
        #cdef Data data = self.data

        # fill label_count_total and weighted_n_node_samples
        cdef SIZE_t n_outputs  = data.n_outputs
        cdef SIZE_t* n_classes = data.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        
        cdef DOUBLE_t* label_count_total = self.label_count_total
        cdef DOUBLE_t weighted_n_node_samples = 0.0
        
        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef SIZE_t c = 0
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        # clear label_count_total 
        for k in range(n_outputs):
            memset(label_count_total + offset, 0, n_classes[k] * sizeof(DOUBLE_t))
            offset += label_count_stride

        # update class distribution (label_count_total)
        # at the same time, update weighted_n_node_samples
        for p in range(start, end):
            i = samples_win[p]

            if data.sample_weight != NULL:
                w = data.sample_weight[i]

            for k in range(n_outputs):
                c = <SIZE_t> data.y[ i*data.y_stride + k]    # y[i,k] 
                label_count_total[k * label_count_stride + c] += w # label_count_total[k,c] 

            weighted_n_node_samples += w

        self.weighted_n_node_samples = weighted_n_node_samples

    cdef void update(self,
            SIZE_t* samples_win,
            SplitRecord split_record, 
            DTYPE_t* Xf  
            ) nogil:       
        '''
        udpate:
            label_count, array[n_subnodes][n_outputs][max_n_classes]
        '''
        cdef Feature* feature = &self.data.features[split_record.feature_index]
       
        cdef Data data = self.data  # XXX
        cdef SIZE_t start = self.start
        cdef SIZE_t end   = self.end

        cdef DOUBLE_t* label_count = self.label_count 
        cdef SIZE_t feature_stride = self.feature_stride 
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef SIZE_t class_label
        cdef SIZE_t label_index
        cdef SIZE_t p, s_i, f_i, k, 
        cdef DOUBLE_t w

        #feature_value = 0  # feature value, from 0 to feature.n_values-1   
        for p in range(start, end):
            s_i = samples_win[p]
           
            if data.sample_weight != NULL:
                w = data.sample_weight[s_i]
            else:
                w = 1.0

            for k in range(data.n_outputs):
                    
                class_label =<SIZE_t> data.y[ s_i * data.y_stride + k] # y[i,k]
                    
                if feature.type == FEATURE_CONTINUOUS:
                    if Xf[p] < split_record.threshold:
                        f_i = 0
                    else:
                        f_i = 1
                else:
                    f_i = <SIZE_t>Xf[p] # XXX type conversion
                
                # label_count[ f_i, k, label ]
                label_index = f_i*feature_stride + k*label_count_stride + class_label
                label_count[label_index] += w

    # gini
    cdef DOUBLE_t node_impurity(self, DOUBLE_t* label_count, DOUBLE_t wn_samples, DOUBLE_t epsilon) nogil:

        cdef UINT32_t* rand = &self.rand_r_state

        #cdef DOUBLE_t* label_count = label_count 
        wn_samples += noise(epsilon, rand) # XXX

        cdef DOUBLE_t total, gini, count
        cdef SIZE_t k, c
        cdef SIZE_t n_outputs = self.data.n_outputs 
        cdef SIZE_t* n_classes = self.data.n_classes 
        cdef SIZE_t label_count_stride = self.label_count_stride

        total = 0.0
        for k in range(n_outputs):
            gini = 0.0

            for c in range(n_classes[k]):
                count = label_count[c] + noise(epsilon, rand) # XXX
                gini += (count*count) / wn_samples 
            
            total += gini
            label_count += label_count_stride 

        return total / n_outputs 


    cdef DOUBLE_t improvement(self, DOUBLE_t* wn_subnodes_samples, SIZE_t n_subnodes, DOUBLE_t epsilon) nogil:
        '''
        calculate improvement based on class distribution
        '''
        cdef DOUBLE_t* label_count = self.label_count
       
        cdef DOUBLE_t improvement = 0

        cdef SIZE_t i = 0
        # sum up node_impurity of each subset 
        for i in range(n_subnodes):
            label_count += self.feature_stride 
            improvement += self.node_impurity(label_count, wn_subnodes_samples[i], epsilon) # XXX

        return improvement

    cdef void node_value(self, DOUBLE_t* dest) nogil:
        '''
        return class distribution of node, i.e. label_count_total
        '''
        cdef SIZE_t n_outputs = self.data.n_outputs
        cdef SIZE_t* n_classes = self.data.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef DOUBLE_t* label_count_total = self.label_count_total 

        cdef SIZE_t k
        for k in range(n_outputs):
            memcpy(dest, label_count_total, n_classes[k] * sizeof(DOUBLE_t))

            dest += label_count_stride
            label_count_total += label_count_stride 

# ===========================================================
# Splitter 
# ===========================================================
cdef inline void _init_split(SplitRecord* self) nogil:
    self.feature_index = 0
    self.threshold = 0.0
    self.improvement = -INFINITY

    self.n_subnodes = 0
    self.n_subnodes_samples = NULL
    self.wn_subnodes_samples = NULL

cdef class Splitter:


    def __cinit__(self,
                    Criterion criterion,
                    SIZE_t max_candid_features,
                    object random_state):
        self.criterion = criterion
        # self.data = None #XXX
        
        self.random_state = random_state 
        self.rand_r_state = random_state.randint(0, RAND_R_MAX)

        self.samples_win = NULL
        self.start = 0
        self.end   = 0

        self.features_win = NULL
        self.n_features = 0

        self.feature_values  = NULL # tempory array
        self.max_candid_features = max_candid_features
   
        self.debug = True
    
    def __dealloc__(self):
        free(self.samples_win)
        free(self.features_win)
        free(self.feature_values)

    cdef void init(self, 
            Data data) except *:
        ''' set data
            alloc samples_win, features_win, feature_values
        '''
        # set samples window
        cdef SIZE_t n_samples = data.n_samples
        cdef SIZE_t* samples_win  = safe_realloc(&self.samples_win, n_samples)

        cdef SIZE_t i
        for i in range(n_samples):
            samples_win[i] = i

        # set features window
        cdef SIZE_t  n_features = data.n_features
        cdef SIZE_t* features_win = safe_realloc(&self.features_win, n_features)
        for i in range(n_features):
            features_win[i] = i

        safe_realloc(&self.feature_values, n_samples) 

        # set data
        self.data = data

    cdef void node_reset(self, SIZE_t start, SIZE_t end, 
                    DOUBLE_t* weighted_n_node_samples) nogil:
        ''' call once for each node 
            set start, end, 
            return weighted_n_node_samples'''        

        self.start = start
        self.end   = end

        self.criterion.init(self.data,
                            self.samples_win,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

    cdef void _choose_split_point(self) nogil:
        pass

    cdef void _choose_split_feature(self,
            SplitRecord* best,
            SplitRecord* split_records, 
            SIZE_t size, 
            DOUBLE_t epsilon) nogil:
        pass

    cdef void node_split(self, 
            SplitRecord* split_record, 
            SIZE_t* n_node_features,
            DOUBLE_t epsilon):
        ''' 
            Calculate:
                best feature for split,
                best split point (for continuous feature)
        '''
        # create and init split records
        cdef SplitRecord* feature_records = <SplitRecord*> calloc(n_node_features[0], sizeof(SplitRecord))
        cdef SIZE_t i
        for i in range(n_node_features[0]):
            _init_split(&feature_records[i])

        cdef SplitRecord current, best
        _init_split(&current)
        _init_split(&best)

        cdef Data data = self.data
        cdef SIZE_t* samples_win = self.samples_win

        cdef Feature* feature
        cdef SIZE_t* features_win = self.features_win 
        cdef SIZE_t f_i = n_node_features[0]-1
        cdef SIZE_t f_j = 0

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t start = self.start,
        cdef SIZE_t end   = self.end
        cdef DOUBLE_t w   = 0.0

        cdef SIZE_t* n_subnodes_samples
        cdef DOUBLE_t* wn_subnodes_samples

        while f_j <= f_i :
            _init_split(&current)
            current.feature_index = features_win[f_j]
            feature = &data.features[current.feature_index]

            # copy to Xf
            for p in range(start, end):
                # Xf[p] = X[sample_index, feature_index]
                Xf[p] = data.X [     
                        samples_win[p]        * data.X_sample_stride  
                     +  current.feature_index * data.X_feature_stride ]
            
            # TODO use _tree.py sort() 
            # XXX type conversion
            sort( Xf+start, samples_win+start, end-start)
    

            # if constant feature
            if Xf[end-1] <= Xf[start] + FEATURE_THRESHOLD:
                features_win[f_j] = features_win[f_i]
                features_win[f_i] = current.feature_index 
                f_i -= 1
                # goes to next candidate feature
            
            # not constant feature
            else:                
                # if continuous feature
                if feature.type == FEATURE_CONTINUOUS:
                    raise ValueError("Warning, node_split not support continuous feature") 
                    current.n_subnodes = 2
                    
                    # TODO set threshold, 
                    self._choose_split_point(&current)
                    # set  n_subnodes_samples
                    # set wn_subnodes_samples
                    
                else:
                    current.n_subnodes = feature.n_values
                    
                    n_subnodes_samples  = safe_realloc(&current.n_subnodes_samples,  current.n_subnodes)
                    wn_subnodes_samples = safe_realloc(&current.wn_subnodes_samples, current.n_subnodes)
                    
                    memset(n_subnodes_samples, 0, sizeof(SIZE_t)*current.n_subnodes)
                    memset(wn_subnodes_samples,0, sizeof(DOUBLE_t)*current.n_subnodes)


                    for i in range(start,end):
                        if data.sample_weight == NULL:
                            w = 1.0
                        else:
                            w = data.sample_weight[ i ]

                        # XXX type convert
                        n_subnodes_samples [ <SIZE_t>Xf[i] ] += 1 
                        wn_subnodes_samples[ <SIZE_t>Xf[i] ] += w
                    if self.debug:
                        printf("distribution:\n")
                        for i in range(current.n_subnodes):
                            printf("%u ",n_subnodes_samples[i])
                        printf("\n")
                 
                self.criterion.update(samples_win, current, Xf)
                current.improvement = self.criterion.improvement(current.wn_subnodes_samples, current.n_subnodes, epsilon)

                feature_records[f_j] = current
                if current.improvement > best.improvement:
                    best = current

                f_j += 1

        self._choose_split_feature(&best, 
                            feature_records, 
                            f_i, 
                            epsilon)

        split_record[0]    = best
        n_node_features[0] = f_i+1 

    cdef void node_value(self,DOUBLE_t* dest) nogil:
        self.criterion.node_value(dest)

cdef class LapSplitter(Splitter):

    cdef void _choose_split_point(self) nogil:
        with gil:
            raise("Laplace mech doesn't support continuous feature") 
        
    # choose the best
    cdef void _choose_split_feature(self,
                SplitRecord* best, 
                SplitRecord* records,
                SIZE_t size,
                DOUBLE_t epsilon) nogil:
        ''' Choose the best split feature ''' 
        cdef SIZE_t i, max_index = 0
        cdef DOUBLE_t max_improvement = -INFINITY

        for i in range(size):
            if records[i].improvement > max_improvement:
                max_index = i
                max_improvement = records[i].improvement 
        
        best[0] = records[max_index]

cdef class ExpSplitter(Splitter):

    cdef void _choose_split_point(self) nogil:
        pass
    
    cdef void _choose_split_feature(self,
                SplitRecord* best, 
                SplitRecord* records,
                SIZE_t size,
                DOUBLE_t epsilon) nogil:

        cdef UINT32_t* rand = &self.rand_r_state

        cdef SIZE_t index = 0      
        index = draw_from_exponential_mech(
                epsilon,
                records, 
                size,
                self.criterion.sensitivity,
                rand)
        
        best[0] = records[index]

cdef class DataObject:
    
    def __cinit__(self, 
                    np.ndarray[DTYPE_t, ndim=2] X,
                    np.ndarray[DOUBLE_t, ndim=2, mode="c"] y, 
                    meta, np.ndarray sample_weight):
        
        cdef SIZE_t i
        cdef SIZE_t k
        
        cdef SIZE_t n_samples
        cdef SIZE_t n_features
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        cdef DOUBLE_t weighted_n_samples = 0.0
        if sample_weight is not None:
            for i in range(n_samples):
                weighted_n_samples += sample_weight[i]
        # y
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y, (-1,1))
       
        # class
        cdef SIZE_t n_outputs = y.shape[1]
        y = np.copy(y) # XXX why copy?
        # cdef SIZE_t* classes = []
        cdef SIZE_t* n_classes = <SIZE_t*>calloc(n_outputs, sizeof(SIZE_t))
        cdef SIZE_t max_n_classes = 0
        for k in range( n_outputs ):
            classes_k, y[:,k] = np.unique(y[:,k], return_inverse=True)
            # classes.append(classes_k)
            n_classes[k] = classes_k.shape[0]
            if classes_k.shape[0] > max_n_classes:
                max_n_classes = classes_k.shape[0]
        # n_classes = np.array( n_classes, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        
        # features 
        cdef Feature* features = <Feature*>calloc(n_features,sizeof(Feature))
        for i in range(n_features):
            features[i].type = FEATURE_DISCRETE
            features[i].n_values = meta.features_[i].n_values
        
        cdef SIZE_t max_n_feature_values = 0
        cdef SIZE_t n_continuous_features = 0
        for i in range(n_features):
            if features[i].n_values > max_n_feature_values:
                max_n_feature_values = features[i].n_values
            if features[i].type == FEATURE_CONTINUOUS:
                n_continuous_features += 1

        # set data
        cdef Data data 

        data.X = <DTYPE_t*> X.data
        data.y = <DOUBLE_t*> y.data
        if sample_weight is None:
            data.sample_weight = NULL
        else:
            data.sample_weight = <DOUBLE_t*> sample_weight.data

        data.X_sample_stride  = <SIZE_t> X.strides[0]/<SIZE_t> X.itemsize
        data.X_feature_stride = <SIZE_t> X.strides[1]/<SIZE_t> X.itemsize
        data.y_stride = <SIZE_t> y.strides[0] /<SIZE_t> y.itemsize
        
        data.n_samples = n_samples
        data.weighted_n_samples = weighted_n_samples

        data.n_features = n_features
        data.features = features
        data.max_n_feature_values = max_n_feature_values
        data.n_continuous_features = n_continuous_features

        # classes
        data.n_outputs = n_outputs
        # data.classes   = classes
        data.n_classes = n_classes
        data.max_n_classes = max_n_classes

        self.data = data

cdef class NBTreeBuilder:

    def __cinit__(self, 
                    SIZE_t diffprivacy_mech,
                    DOUBLE_t budget,
                    Splitter splitter,
                    SIZE_t max_depth,
                    SIZE_t max_candid_features,
                    object random_state
                    ):

        self.diffprivacy_mech = diffprivacy_mech
        self.budget = budget

        self.splitter = splitter

        self.max_depth = max_depth  # verified by Classifier
        self.max_candid_features = max_candid_features # verified

        # self.tree = None    # XXX ??? pass parameter from python
        # self.data = None
        
        self.random_state = random_state
        self.rand_r_state = random_state.randint(0, RAND_R_MAX)
        

    # cpdef build(self):
    cpdef build(self,
                Tree    tree,
                DataObject dataobject,
                int     debug):
       
        printf("Get into build function\n")

        cdef Data data = dataobject.data
        cdef UINT32_t* rand = &self.rand_r_state
        cdef Splitter splitter = self.splitter
       
        # set parameter for building tree 
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t max_candid_features = self.max_candid_features

        printf("Init capacity of tree\n")

        # Initial capacity of tree
        cdef SIZE_t init_capacity
        if max_depth <= 10:
            init_capacity = (2 ** (max_depth + 1)) - 1
        else:
            init_capacity = 2047
        printf("Tree init_capacity: %d \n", init_capacity) 
        tree._resize(init_capacity)
        self.tree = tree

        printf("Start to set diffprivacy parameter\n")
        # set parameter for diffprivacy
        cdef DOUBLE_t budget = self.budget
        cdef SIZE_t diffprivacy_mech = self.diffprivacy_mech
        cdef DOUBLE_t epsilon_per_depth = 0.0
        cdef DOUBLE_t epsilon_per_action= 0.0   
        if diffprivacy_mech == LAP_DIFF_PRIVACY_MECH:
            epsilon_per_depth = budget/(max_depth+1)
            epsilon_per_action = epsilon_per_depth/2.0
        elif diffprivacy_mech == EXP_DIFF_RPIVACY_MECH:
            epsilon_per_action = budget/( (2+data.n_continuous_features)*max_depth + 2)
        else:
            epsilon_per_action = NO_DIFF_PRIVACY_BUDGET
        
        printf("espilon per action is %f\n", epsilon_per_action)

        # ===================================================
        # recursively depth first build tree 

        splitter.init(data) # set samples_win, features_win
        cdef SplitRecord split_record 

        cdef SIZE_t max_depth_seen = -1 # record the max depth ever seen
        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t start_i
        cdef SIZE_t end_i
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef SIZE_t index
        cdef SIZE_t n_node_features
        cdef SIZE_t n_node_samples
        cdef DOUBLE_t noisy_n_node_samples
        cdef DOUBLE_t weighted_n_node_samples
        cdef SIZE_t node_id

        # init Stack structure
        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record
        # root node record
        #stack_record.start = 0
#        stack_record.end   = data.n_samples 
#        stack_record.depth = 0
#        stack_record.parent=_TREE_UNDEFINED
#        stack_record.index = 0
#        stack_record.n_node_features  = data.n_features
        # push root node into stack
        rc = stack.push(0,              # start
                        data.n_samples, # end
                        0,              # depth
                        _TREE_UNDEFINED,# parent
                        0,              # index
                        data.n_features)# n_node_features

        if rc == -1:
            raise MemoryError()

        cdef SIZE_t n_loop = 0
        with nogil:
            while not stack.is_empty():

                printf("loop %d begin\n", n_loop)
                n_loop += 1

                stack.pop(&stack_record)
                start = stack_record.start
                end   = stack_record.end
                depth = stack_record.depth
                parent= stack_record.parent
                index = stack_record.index
                n_node_features = stack_record.n_node_features 

                n_node_samples = end - start
                noisy_n_node_samples = <DOUBLE_t>n_node_samples + noise(epsilon_per_action, rand) # XXX
                # reset class distribution based on this node
                # node_reset
                #       fill label_total_count,
                #       calculate weighted_n_node_samples
                printf("node_reset: from %u to %u\n", start, end)
                splitter.node_reset(start, end, &weighted_n_node_samples)

                # if leaf
                if (  depth >= max_depth 
                   or n_node_features <= 0 
                   or noisy_n_node_samples/(data.max_n_feature_values*data.max_n_classes) < sqrt(2.0)/epsilon_per_action ) : # xxx
                    if debug:
                        printf("becomes a leaf node\n")
                    # leaf node
                    node_id = tree._add_node(
                            parent,
                            index,
                            True,      # leaf node
                            NO_FEATURE,
                            NO_THRESHOLD,
                            0,                      # no children
                            n_node_samples,
                            weighted_n_node_samples # xxx
                            )

                    # XXX
                    # tree.nodes[node_id].values = <DOUBLE_t*>calloc( data.n_outputs * data.max_n_classes, sizeof(DOUBLE_t))

                    # store class distribution into node.values
                    splitter.node_value(tree.value+node_id*tree.value_stride)
                    
                    # add noise to the class distribution
                    noise_distribution(epsilon_per_action, tree.value+node_id*tree.value_stride, data, rand)
                else:       
                    if debug:
                        printf("becomes a inner node\n")
                    # inner node
                    # choose split feature
                    # XXX: refine node_split

                    with gil:
                        splitter.node_split( &split_record, &n_node_features, epsilon_per_action )
                
                    node_id = tree._add_node(
                            parent,
                            index,
                            False,     # not leaf node
                            split_record.feature_index,
                            split_record.threshold,
                            split_record.n_subnodes,  # number of children
                            n_node_samples, 
                            weighted_n_node_samples # XXX
                            )

                    if debug:
                        printf("New inner Node\n parent\t%u,\n index\t%u,\n feature\t%u,\n n_children\t%u,\n n_node_samples\t%u,\n weighted_node_samples\t%f\n\n",
                                parent, index, split_record.feature_index, split_record.n_subnodes,
                                n_node_samples, weighted_n_node_samples)

                    # push children into stack
                    # split_feature = data.features[split_record.feature_index]   

                    start_i = 0
                    end_i   = 0
                    for index in range(split_record.n_subnodes):
                        start_i = end_i
                        end_i   = end_i + split_record.n_subnodes_samples[index]
                        
                        rc = stack.push(
                            start_i,    # start pos
                            end_i,      # end pos
                            depth+1,    # depth of this new node
                            node_id,    # child's parent id
                            index,      # the index
                            n_node_features 
                            )
                        if debug:
                            printf("Push new into stack\n start\t%u,\n end\t%u,\n depth\t%u,\n parent\t%u,\n index\t%u,\n n_node_features\t%u\n\n",
                                start_i,end_i,depth+1,node_id,index,n_node_features)

                        if rc == -1:
                            break
                            # raise MemoryError()
                            
                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        
        if rc == -1:
            raise MemoryError()
            # TODO: prune

        printf("Finished building the tree\n")        
 
cdef class Tree:
    ''' Array based no-binary decision tree'''
    property n_classes:
        def __get__(self):
            # it's small; copy for memory safety
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs).copy()

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
            DataObject dataobject):
                #Feature*       features,
                #SIZE_t         n_features,
                #np.ndarray[SIZE_t, ndim=1]  n_classes,
                #SIZE_t         n_outputs):

        cdef Data data = dataobject.data
        cdef SIZE_t* n_classes = data.n_classes

        self.features = data.features    # XXX useful?
        
        self.n_features = data.n_features
        self.n_outputs  = data.n_outputs
        self.n_classes  = NULL
        safe_realloc(&self.n_classes, self.n_outputs)

        self.max_n_classes = data.max_n_classes
        self.value_stride = self.n_outputs * self.max_n_classes

        cdef SIZE_t k
        for k in range(self.n_outputs):
            self.n_classes[k] = n_classes[k]

        self.node_count = 0
        self.capacity   = 0
        self.nodes      = NULL
        self.value      = NULL
        
        self.max_depth  = 0
    
    def __dealloc__(self):
        free(self.n_classes)
        free(self.nodes)
        free(self.value)

    cdef void _resize(self, SIZE_t capacity):
        if self._resize_c(capacity) != 0:
            raise MemoryError()
    
    ''' 
        capacity by default = -1, means double the capacity of the inner struct
    '''
    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) nogil:
       
        printf("Get in resize c\n")
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity
        printf("XXX \n") 
 

        # XXX no safe_realloc here because we need to grab the GIL
        # realloc self.nodes
        cdef void* ptr = realloc(self.nodes, capacity * sizeof(Node))
        if ptr == NULL:
            return -1
        self.nodes = <Node*> ptr
        
        printf("XXX \n") 
        ptr = realloc(self.value,
                      capacity * self.value_stride * sizeof(double))
        if ptr == NULL:
            return -1
        self.value = <double*> ptr
        
        printf("XXX \n") 
 
        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 
                    0,
                   (capacity - self.capacity) * self.value_stride * sizeof(double))
        
        printf("XXX \n") 
 
        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity
        printf("XXX \n") 
 
        self.capacity = capacity
        printf("XXX \n") 
        printf("XXX get out of resize c \n") 

        return 0

    cdef SIZE_t _add_node(self, 
                          SIZE_t parent, 
                          SIZE_t index,
                          bint   is_leaf,
                          SIZE_t feature, 
                          DOUBLE_t threshold, 
                          SIZE_t n_children,
                          SIZE_t n_node_samples, 
                          DOUBLE_t weighted_n_node_samples
                          ) nogil:
        """Add a node to the tree.
        The new node registers itself as the child of its parent.
        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return <SIZE_t>(-1)

        cdef Node* node = &self.nodes[node_id]
                
        node.parent = parent
        node.is_leaf = is_leaf

        node.feature = feature
        node.threshold = threshold
        # node.impurity = impurity
        
        node.n_children = n_children 
        if is_leaf and n_children == 0:
            node.children = NULL
        else:
            node.children = <SIZE_t*>calloc(n_children, sizeof(SIZE_t))
            if node.children == NULL: # XXX error
                return <SIZE_t>(-1)

        if parent != _TREE_UNDEFINED:
            if self.nodes[parent].n_children <= index:
                with gil:
                    raise ValueError("child's index %d is greater than parent's n_classes %d"%(index, self.nodes[parent].n_children))
            self.nodes[parent].children[index]=node_id
        
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        self.node_count += 1
        return node_id

    cpdef np.ndarray predict(self, np.ndarray[DTYPE_t, ndim=2] X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out
    
    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        
        cdef np.ndarray arr  # shape is [0] x [1] x [2]
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
        arr = PyArray_NewFromDescr( np.ndarray, 
                                    <np.dtype> NODE_DTYPE, # XXX
                                    1, shape,
                                    strides, <void*> self.nodes,
                                    np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cpdef np.ndarray apply(self, np.ndarray[DTYPE_t, ndim=2] X):
        
        cdef SIZE_t n_samples = X.shape[0]
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        cdef np.ndarray[SIZE_t] node_ids = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* id_data = <SIZE_t*> node_ids.data

        with nogil:
            for i in range(n_samples):
                node = self.nodes

                while not node.is_leaf:
                    if self.features[node.feature].type == FEATURE_CONTINUOUS:
                        if X[i, node.feature] < node.threshold:
                            node = &self.nodes[node.children[0]]
                        else:
                            node = &self.nodes[node.children[1]]
                    else:
                        node = &self.nodes[node.children[ <SIZE_t>X[i, node.feature] ]]

                id_data[i] = <SIZE_t>( node - self.nodes )

        return node_ids

# =========================================================================
# Utils
# ========================================================================

# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DOUBLE_t*) # added
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

cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
    """Encapsulate data into a 1D numpy array of intp's."""
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> size
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data)

# ==================================================================
# Utils for diffprivacy
# ==================================================================
cdef DOUBLE_t laplace(DOUBLE_t b, UINT32_t* rand) nogil except -1:
    # No diffprivacy
    if b <= 0.0:
        return 0.0

    cdef DOUBLE_t uniform = rand_double(rand)-0.5 # gil
    if uniform > 0.0:
        return -b*log(1.0-2*uniform)
    else:
        return +b*log(1.0+2*uniform) 

cdef DOUBLE_t noise(DOUBLE_t epsilon, UINT32_t* rand) nogil:
    if epsilon == NO_DIFF_PRIVACY_BUDGET:
        return 0.0

    return laplace(1.0/epsilon, rand)

cdef void noise_distribution(DOUBLE_t epsilon, DOUBLE_t* dest, Data data, UINT32_t* rand) nogil:

    cdef SIZE_t k = 0
    cdef SIZE_t i = 0
    
    cdef SIZE_t n_outputs = data.n_outputs
    cdef SIZE_t* n_classes = data.n_classes
    cdef SIZE_t stride = data.max_n_classes
    cdef SIZE_t zero_cnt = 0

    for k in range(n_outputs):
        
        for i in range(n_classes[k]):
            dest[i] += noise(epsilon, rand)

        zero_cnt = 0
        for i in range(n_classes[k]):
            if dest[i] <= 0.0:
               dest[i] = 0.0
               zero_cnt += 1
        if zero_cnt ==  n_classes[k]:
            dest[rand_int(n_classes[k], rand)] = 1.0

        dest += stride


cdef SIZE_t draw_from_distribution(DOUBLE_t* dist, SIZE_t size, UINT32_t* rand) nogil except -1:
    """ numbers in arr should be greater than 0 """

    cdef DOUBLE_t total = 0.0
    cdef DOUBLE_t point = 0.0 
    cdef DOUBLE_t current = 0.0 

    cdef SIZE_t i = 0
    for i in range(size):
        
        if dist[i] < 0.0:
            with gil:
                raise("numbers in dist should be greater than 0, but dist[",i,"]=", dist[i], "is not.")
            #return -1

    # if numbers in arr all equal to 0
    if total == 0.0:
        return rand_int(size, rand)

    dist[i] = dist[i]/total

    point = rand_double(rand)

    i = 0
    for i in range(size):
        current += dist[i]
        if current > point:
            return i

    return size-1


cdef SIZE_t draw_from_exponential_mech(double epsilon, SplitRecord* records, int size, double sensitivity, UINT32_t* rand) nogil except -1:

    cdef double* improvements
    cdef double max_improvement = -INFINITY
    cdef int i
    cdef int ret_idx

    improvements = <double*>calloc(size, sizeof(double))
    
    i = 0
    for i in range(size):
        improvements[i] = records[i].improvement
        if improvements[i] > max_improvement:
            max_improvement = improvements[i]

    # rescale from 0 to 1
    i = 0
    for i in range(size):
        improvements[i] -= max_improvement
        improvements[i] = exp(improvements[i]*epsilon/(2*sensitivity))

    ret_idx = draw_from_distribution(improvements, size, rand)

    free(improvements)

    return ret_idx


