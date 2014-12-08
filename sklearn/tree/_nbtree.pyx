cdef struct Feature:

    cdef int type       # continuous or discrete feature
    
    cdef int n_values   # the number of distinct value of the feature
                        # for continuous feature, = 2  

    cdef double max     # for continuous feature only
    cdef double min     # for continuous feature only

cdef struct SplitRecord:
    SIZE_t feature_index    # the index of the feature to split
    SIZE_t threshold        # for continuous feature only
    DOUBLE_t improvement    # the improvement by selecting this feature
   
    SIZE_t  n_subnodes
    SIZE_t* n_subnodes_samples
    SIZE_t* wn_subnodes_samples 

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

cdef class Data:

    DOUBLE_t* X

    DOUBLE_t* y
    SIZE_t    y_stride
    SIZE_t    n_outputs
    np.ndarray[SIZE_t, ndim=1] n_classes 
    SIZE_t    max_labels    # max(number of distinct class label)

    SIZE_t    max_feature_values # max(number of distint values of all feature)

    SIZE_t  n_samples 
    SIZE_t  weighted_n_samples 

######################################################################
from libc.stdio cimport calloc, free, realloc
from libc.string cimport memcpy, memset
from libc.math   cimport log as ln
from cpython cimport Py_INCREF, PyObject

from sklearn.tree._nbutils cimport Stack, StackRecord

import numpy as np
cimport numpy as np
np.import_array() # XXX

# ====================================================================
# Criterion
# ====================================================================

cdef class Criterion:

    cdef SIZE_t start
    cdef SIZE_t end
    cdef SIZE_t pos

    Feature feature 

    double* label_count_total   # shape[n_outputs][max_n_classes]
    double* label_count         # shape[max_n_feature_values][n_outputs][max_n_classes] 
    cdef SIZE_t label_count_stride
    cdef SIZE_t feature_stride 

    def __cinit__(self, Data data):
        '''
            allocate:
                label_count,
                label_count_total

            set:
                label_count_stride, = max_n_classes 
                feature_stride,     = max_n_classes * n_outputs 
        '''
        # self.data = data

        self.samples = NULL

        self.start = 0
        self.end   = 0

        self.label_count_stride = data.max_n_classes 

        cdef SIZE_t feature_stride = n_outputs * label_count_stride
        self.label_count_total 
            = <double*> calloc(feature_stride, sizeof(double))
        self.label_count 
            = <double*> calloc(feature_stride*data.max_n_feature_values, sizeof(double))

        self.feature_stride = feature_stride

        if self.label_count_total == NULL 
        or self.label_count == NULL:
            raise MemoryError()

    def init(self, 
                   Data      data,
                   SIZE_t*   samples, 
                   SIZE_t    start, 
                   SIZE_t    end,
                   SIZE_t    weighted_n_node_samples
                   ) nogil:
        '''
        For one node, called once,
        update class distribution in this node
            
        fill: 
            label_count_total, shape[n_outputs][max_n_classes]    
        update:
            weighted_n_node_samples  
        '''
        # cdef UINT32_t* random_state = &self.rand_r_state
        
        # Initialize fields
        # self.sample_weight = sample_weight
        # self.samples  = samples
        
        self.start    = start
        self.end      = end
        self.n_node_samples = end - start
      
        # XXX
        Data data = data

        # fill label_count_total and weighted_n_node_samples
        cdef SIZE_t n_outputs  = data.n_outputs
        cdef SIZE_t* n_classes = data.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        
        cdef double* label_count_total = self.label_count_total
        cdef double weighted_n_node_samples = 0.0
        
        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef SIZE_t c = 0
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        # clear label_count_total 
        for k in range(n_outputs):
            memset(label_count_total + offset, 0, n_classes[k] * sizeof(double))
            offset += label_count_stride

        # update class distribution (label_count_total)
        # at the same time, update weighted_n_node_samples
        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                c = <SIZE_t> y[ i*y_stride + k]    # y[i,k] 
                label_count_total[k * label_count_stride + c] += w # label_count_total[k,c] 

            weighted_n_node_samples += w

        self.weighted_n_node_samples = weighted_n_node_samples


    def update(self,
            SplitRecord split_record, 
            double* Xf      # only for continuous feature
            ):       
        '''
        udpate:
            label_count, array[n_subnodes][n_outputs][max_n_classes]
        '''
        cdef Feature* feature = &self.data.features[split.feature_index]
        
        cdef SIZE_t start = self.start
        cdef SIZE_t end   = self.end

        cdef SIZE_t feature_stride = self.feature_stride 
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef SIZE_t class_label
        cdef SIZE_t label_index
        cdef SIZE_t p, s_i, f_i, k, 
        cdef DOUBLE_t w

        #feature_value = 0  # feature value, from 0 to feature.n_values-1   
        for p in range(start, end):
            s_i = samples_win[p]
           
            if self.data.sample_weight != NULL:
                w = self.data.sample_weight[s_i]
            else:
                w = 1.0

            for k in range(n_outputs):
                    
                class_label =<SIZE_t> data.y[ s_i * data.y_stride + k] # y[i,k]
                    
                if feature.type == FEATURE_CONTINUOUS:
                    if Xf[p] < split_record.threshold:
                        f_i = 0
                    else:
                        f_i = 1
                else:
                    f_i = Xf[p]

                label_index = f_i * feature_stride            # label_count[ fvalue, k, label ]
                            + k   * label_count_stride
                            + class_label

                label_count[label_index] += w

    # gini
    def double node_impurity(self, double* label_count, SIZE_t wn_samples):

        cdef double* label_count = label_count 
        cdef SIZE_t wn_samples = wn_samples # + laplace(epsilon_per_action)

        cdef double total, gini, count
        cdef SIZE_t k, c
        cdef SIZE_t n_outputs = self.data.n_outputs 
        cdef SIZE_t* n_classes = self.data.n_classes 
        cdef SIZE_t label_count_stride = self.label_count_stride

        total = 0.0
        for k in range(n_outputs):
            gini = 0.0

            for c in range(n_classes[k]):
                count = label_count[c] # + laplace(epsilon_per_action)
                gini += (count*count) / wn_samples 
            
            total += gini
            label_count += label_count_stride 

        return total / n_outputs 


    def double improvement(self, SIZE_t* wn_subnodes_samples, SIZE_t n_subnodes):
        '''
        calculate improvement based on class distribution
        '''
        cdef double* label_count = self.label_count
       
        cdef double improvements = 0

        cdef SIZE_t i = 0
        # sum up node_impurity of each subset 
        for i in range(n_subnodes):
            label_count += self.feature_stride 
            improvement += self.node_impurity(label_count, wn_subnodes_samples[i])

        return improvement

    def node_value(self, double* dist):
        '''
        return class distribution of node, i.e. label_count_total
        '''
        cdef SIZE_t n_outputs = self.data.n_outputs,
                    n_classes = self.data.n_classes
                    label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total 


        cdef SIZE_t k
        for k in range(n_outputs):
            memcpy(dest, label_count_total, n_classes[k] * sizeof(double))

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
    self.n_subnodes_samples = None
    self.wn_subnodes_samples = None

cdef class Splitter:

    cdef public Criterion criterion
    cdef Data data
    cdef Object random

    cdef SIZE_t* samples_win
    cdef SIZE_t  start, end

    cdef SIZE_t* features_win
    cdef SIZE_t  n_features
    
    cdef DOUBLE_t feature_values        # .temp

    cdef SIZE_t max_candid_features     # reserved

    def __cinit__(self,
                    Criterion criterion,
                    SIZE_t max_candid_features,
                    object random_state):

        self.criterion = criterion
       
        self.samples_win = NULL
        
        self.features_window = NULL
        self.feature_values  = NULL # tempory array
        self.max_candid_features = max_candid_features
        
        self.random = ... 
    
    cdef init(self, 
            Data data):
            #np.ndarray[DTYPE_t, ndim=2] X,
            #np.ndarray[double,  ndim=2] y,
            #Feature* features):
        '''
        set samples_win
        set features_win 
        set feature_values
        '''
        # set samples window
        cdef SIZE_t n_samples = data.n_samples
        cdef SIZE_t* samples_win  = safe_realloc(&self.samples_win, n_samples)

        cdef SIZE_t i
        for i in range(n_samples):
            samples_win[i] = i

        # set features window
        cdef SIZE_t  n_features = data.n_features
        cdef SIZE_t* features_win 
                = safe_realloc (&self.features_win, n_features)
        for i in range(n_features):
            features_win[i] = i

        safe_realloc(&self.feature_values, n_samples)

        # set data
        self.data = data

    cdef node_reset(self, SIZE_t start, SIZE_t end, 
                    double* weighted_n_node_samples) nogil:
        ''' call once for each node'''        

        self.start = start
        self.end   = end

        self.criterion.init(
                            self.data,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

    cdef _choose_split_point():
        pass

    cdef _choose_split_feature():
        pass

    cdef node_split(self, 
            double epsilon,
            SplitRecord* split_record, 
            SIZE_t n_node_features):
        ''' 
            Calculate:
                best feature for split,
                best split point (for continuous feature)
        '''
        # create and init split records
        cdef SplitRecord* feature_records 
                = <SplitRecord*> calloc(n_node_features, sizeof(SplitRecord))
        cdef SIZE_t i
        for i in range(n_features):
            _init_split(&feature_records[i])

        cdef SplitRecord current, best
        _init_split(&current)
        _init_split(&best)

        cdef Data data = self.data
        cdef SIZE_t samples_win = self.samples_win

        cdef Feature* feature
        cdef SIZE_t* features_win = self.features_win 
        cdef SIZE_t f_i = n_node_features-1
                    f_j = 0

        cdef DOUBLE_t* Xf = self.feature_values
        cdef SIZE_t start = self.start,
                    end   = self.end
    

        while f_j <= f_i :
            _init_split(current)
            current.feature_index = features_window[f_j]
            feature = &data.features[current.feature_index]

            # copy to Xf
            for p in range(start, end):
                # Xf[p] = X[sample_index, feature_index]
                Xf[p] = data.X [     
                        samples_win[p]        * data.X_sample_stride  
                     +  current.feature_index * data.X_feature_stride ]
            
            # TODO use _tree.py sort() 
            sort(Xf+start, samples+start, end-start)

            # if constant feature
            if Xf[end-1] <= Xf[start] + FEATURE_THRESHOLD:
                features_win[f_j] = features_win[f_i]
                features_win[f_i] = current.feature_index 
                f_i --
                # goes to next candidate feature
            
            # not constant feature
            else:                
                # if continuous feature
                if feature.type == FEATURE_CONTINUOUS:
                    raise("Warning, node_split not support continuous feature") 
                    current.n_subnodes = 2
                    
                    # TODO set threshold, 
                    _choose_split_point(&current)
                
                    # set  n_subnodes_samples
                    # set wn_subnodes_samples
                    
                else:
                    current.n_subnodes = feature.n_values

                    n_subnodes_samples  = <SIZE_t*> safe_realloc(current.n_subnodes_samples,  feature.n_values)
                    wn_subnodes_samples = <SIZE_t*> safe_realloc(current.wn_subnodes_samples, feature.n_values)
                    
                    for i in range(start,end):
                        if data.samples_weight == NULL:
                            w = 1.0
                        else:
                            w = data.sample_weight[ i ]

                        n_subnodes_samples[ Xf[i] ] ++
                        wn_subnodes_samples[ Xf[i] ] += w

                # XXX
                self.criterion.update(current)
                current.improvement = self.criterion.improvement(current.wn_subnodes_samples, current.n_values)

                feature_records[f_j] = current
                if current.improvement > best.improvement
                    best = current

                f_j ++

        _choose_split_feature(&best, 
                            feature_records, 
                            f_i, 
                            epsilon_per_action)

        split_record[0]    = best
        n_node_features[0] = f_i+1 

cdef class LapSplitter(Splitter):

    cdef _choose_split_point():
        raise("Laplace mech doesn't support continuous feature") 

    # choose the best
    cdef _choose_split_feature(
                SplitRecord* best, 
                SplitRecord* records,
                int size,
                double epsilon):
        ''' Choose the best split feature ''' 
        cdef int i, max_index = 0
        cdef double max_improvement = -INFINITY

        for i in range(size):
            if records[i].improvement > max_improvement:
                max_index = i
                max_improvement = records[i].improvement 
        
        best[0] = records[max_index]

cdef class ExpSplitter(Splitter):

    cdef _choose_split_point():
        pass
    
    cdef _choose_split_feature(
                SplitRecord* best, 
                SplitRecord* records,
                int size,
                double epsilon):

        cdef SIZE_t index = 0      
        index = draw_from_exponential_mech(
                epsilon,
                records, 
                size,
                criterion.sensitivity,
                random)
        
        best[0] = records[index]

cdef class DepthFirstNBTreeBuilder:

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

    cdef __cinit__(int diffprivacy_mech,
                    double budget,
                    Splitter splitter,
                    Tree    tree,
                    SIZE_t max_depth,
                    SIZE_t max_candid_features):

        self.diffprivacy_mech = diffprivacy_mech
        self.budget = budget

        self.splitter = splitter

        self.max_depth = max_depth  # verified by Classifier
        self.max_candid_features = max_candid_features # verified

        self.tree = NULL    # XXX ??? pass parameter from python
        self.data = NULL

    cdef _init_data(self, Data data, np.ndarray X, np.ndarray y, np.ndarray sample_weight=None, Feature* features):

        data.X = X
        data.y = y
        data.sample_weight = sample_weight
        
        data.features = features

    cpdef build(self,
                Tree    tree,
                np.ndarray  X,
                np.ndarray  y,
                np.ndarray  sample_weight=None,
                Feature* features):

        self.data = _init_data(X,y,sample_weight,features)

        cdef Data data = self.data
        cdef Splitter splitter = self.splitter
        cdef Tree tree = tree
        _init_tree(tree, max_depth) # init tree capacity based on max_depth
        
        # set parameter for building tree 
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t max_candid_features = self.max_candid_features

        # set parameter for diffprivacy
        cdef DOUBLE_t budget = self.budget
        cdef double epsilon_per_action   # XXX ?
        printf("espilon per action is %d", epsilon_per_action)

        #######################
        # recursively depth first build tree 

        splitter.init(data) # set samples_win, features_win
        SplitRecord split_record 

        cdef SIZE_t max_depth_seen = -1 # record the max depth ever seen
        cdef SIZE_t start, end,
                    depth,
                    parent, 
                    index,
                    n_node_features
        cdef SIZE_t n_node_samples,
                    noisy_n_node_samples
        cdef SIZE_t node_id

        # init Stack structure
        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record
        # root node record
        stack_record.start = 0
        stack_record.end   = split.n_samples 
        stack_record.depth = 0
        stack_record.parent=_TREE_UNDEFINED
        stack_record.index = 0
        stack_record.n_node_features  = split.n_features
        # push root node into stack
        rc = stack.push(stack_record)

        if rc == -1:
            raise MemoryError()
        with nogil:
            while not stack.is_empty():

                stack.pop(&stack_record)
                start = stack_record.start
                end   = stack_record.end
                depth = stack_record.depth
                parent= stack_record.parent
                index = stack_record.index
                n_node_features = stack_record.n_node_features 

                n_node_samples = end - start
                noisy_n_node_samples = n_node_samples + noise(epsilon_per_action) 
                # reset class distribution based on this node
                # node_reset
                #       fill label_total_count,
                #       calculate weighted_n_node_samples
                splitter.node_reset(start, end, &weighted_n_node_samples)

                # if leaf
                if depth >= max_depth 
                    or n_node_features <= 0
                    or noisy_n_node_samples ... : # XXX
                    
                    # leaf node
                    node_id = tree._add_node(
                            parent,
                            index,
                            True,      # leaf node
                            NO_FEATURE,
                            NO_THRESHOLD,
                            0,                      # no children
                            n_node_samples,
                            weighted_n_node_samples # XXX
                            )

                    # XXX
                    tree.nodes[node_id].values = <double*>calloc( data.n_outputs * data.max_n_classes, sizeof(double))

                    # store class distribution into node.values
                    splitter.node_value(tree.nodes[node_id].values)
                    
                    # add noise to the class distribution
                    noise_distribution(epsilon_per_action, tree.nodes[node_id].values, self.data)
                else:       
                    # inner node
                    # choose split feature
                    # XXX: refine node_split
                    splitter.node_split( &split_record )
                
                    node_id = tree._add_node(
                            parent,
                            index,
                            False,     # not leaf node
                            split_record.feature_index,
                            split_record.threshold,
                            split_record.n_values,  # number of children
                            n_node_samples, 
                            weighted_n_node_samples #XXX
                            )

                    # push children into stack
                    split_feature = data.features[split_record.feature_index]   

                    cdef SIZE_t start_i = 0,
                                end_i   = 0
                    for index in range(split_feature.n_values):
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
                    
                        if rc == -1:
                            raise MemoryError()
                            
                if depth > max_depth_seen:
                    max_depth_seen = depth


            # TODO: prune

cdef class Tree:

    ''' Array based no-binary decision tree

    Attributes
    ----------
    n_features,
    n_outputs,
    n_classes
    max_n_classes

    node_count

    capacity

    max_depth

    nodes : array of Node, shape[ capacity ]
    
    '''

    def __cinit__(self,
                int         n_features
                np.ndarray[SIZE_t, ndim=1]  n_classes,
                int                         n_outputs):

        self.n_features = n_features
        self.n_outputs  = n_outputs
        self.n_classes  = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        
        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        self.node_count = 0
        self.max_depth  = 0
        self.capacity   = 0
        self.nodes      = NULL
    
    def __dealloc__(self):
        free(self.n_classes)
        free(self.nodes)

    def void resize(self, SIZE_t capacity):
        if self._resize_c(capacity) != 0:
            raise MemoryError()
    
    ''' 
        capacity by default = -1, means double the capacity of the inner struct
    '''
    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) nogil:
        
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        # XXX no safe_realloc here because we need to grab the GIL
        # realloc self.nodes
        cdef void* ptr = realloc(self.nodes, capacity * sizeof(Node))
        if ptr == NULL:
            return -1
        self.nodes = <Node*> ptr
        
        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, Node node) nogil:
        pass:

    cpdef np.ndarray predict(self, np.ndarray[DTYPE_t, ndim=2] X):
        
        out = self._get_value_ndarray().take(    
        return out

    cpdef np.ndarray apply(self, np.ndarray[DTYPE_t, ndim=2] X):

        cdef SIZE_t n_samples = X.shape[0]
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        cdef np.ndarray[SIZE_t] node_ids = np.zeros((n_samples,), dtype=np.intp)

        with nogil:
            for i in range(n_samples):
                node = self.nodes

                while node.is_leaf == False:
                    if data.features[node.feature].type == FEATURE_CONTINUOUS:
                        if X[i, node.feature] < node.threshold:
                            node = &self.nodes[node.children[0]]
                        else:
                            node = &self.nodes[node.children[1]]
                    else:
                        node = &self.nodes[node.children[ X[i, node.feature] ]

                node_ids.data[i] = <SIZE_t>( node - self.nodes )

        return node_ids

# Utils

cdef double laplace(double b, UINT32_t* random_state) except -1 with gil:
    # No diffprivacy
    if epsilon <= 0.0:
        return 0.0

    cdef double uniform = rand_double(random_state)-0.5
    if uniform > 0.0:
        return -epsilon*np.log(1.0-2*uniform)
    else:
        return +epsilon*np.log(1.0+2*uniform) 

cdef double noise(double epsilon):
    return laplace(1.0/epsilon, random)

cdef noise_distribution(double epsilon, double* dist, Data data):

    cdef SIZE_t k, i
    
    cdef SIZE_t n_outputs = data.n_outputs
    cdef SIZE_t* n_classes = data.n_classes
    cdef SIZE_t stride = data.max_n_classes
    cdef SIZE_t zero_cnt = 0

    for k in range(n_outputs):
        
        for i in range(n_classes[k]):
            dest[i] += noise(epsilon)

        zero_cnt = 0
        for i in range(n_classes[k]):
            if dest[i] <= 0.0:
               dest[i] = 0.0
               zero_cnt += 1
        if zero_cnt ==  n_classes[k]:
            dest[rand_int(n_classes[k],random_state)] = 1.0

        dest += stride


cdef int draw_from_distribution(double* dist, int size, UINT32_t* random_state) nogil except -1 :
    """ numbers in arr should be greater than 0 """

    cdef double total = 0.0, 
                point = 0.0, 
                current = 0.0 

    cdef int i = 0
    for i in range(size):
        
        if dist[i] < 0.0:
            with gil:
                raise("numbers in dist should be greater than 0, but dist[",i,"]=", dist[i], "is not.")
            #return -1

    # if numbers in arr all equal to 0
    if total == 0.0:
        return rand_int(n,random_state)

    dist[i] = dist[i]/total

    point = rand_double(random_state)

    i = 0
    for i in range(size):
        current += dist[i]
        if current > point:
            return i

    return n-1


cdef int draw_from_exponential_mech(double epsilon, SplitRecord* records, int size, double sensitivity, random):

    cdef double* improvements
    cdef double max_improvement = -INFINITY
    cdef int i
    cdef int ret_idx

    improvements = <double*> calloc(size, sizeof(double))
    
    i = 0
    for i in range(size):
        improvements[i] = records[i].improvement
        if improvements[i] > max_improvement:
            max_improvement = improvements[i]

    with gil:
        # rescale from 0 to 1
        i = 0
        for i in range(size):
            improvements[i] -= max_improvement
            improvements[i] = np.exp(improvements[i]*epsilon/(2*sensitivity))

    ret_idx = draw_from_distribution(improvements, n_split_points, random_state)

    free(improvements)

    return ret_idx


