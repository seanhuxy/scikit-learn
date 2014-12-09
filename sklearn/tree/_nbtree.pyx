from libc.stdio cimport printf
from libc.stdio cimport calloc, free, realloc
from libc.string cimport memcpy, memset
from libc.math   cimport log, exp
from cpython cimport Py_INCREF, PyObject

from sklearn.tree._nbutils cimport Stack, StackRecord

from sklearn.tree._tree cimport safe_realloc, sort # XXX

import numpy as np
cimport numpy as np
from numpy.random import RandomState
np.import_array() # XXX

# ====================================================================
# Criterion
# ====================================================================

cdef class Criterion:

    def __cinit__(self, Data data):
        '''
            allocate:
                label_count,
                label_count_total

            set:
                label_count_stride, = max_n_classes 
                feature_stride,     = max_n_classes * n_outputs 
        '''
        self.data = data

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

    def void init(self, 
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
        Data data = self.data

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
            i = samples_win[p]

            if data.sample_weight != NULL:
                w = data.sample_weight[i]

            for k in range(n_outputs):
                c = <SIZE_t> y[ i*y_stride + k]    # y[i,k] 
                label_count_total[k * label_count_stride + c] += w # label_count_total[k,c] 

            weighted_n_node_samples += w

        self.weighted_n_node_samples = weighted_n_node_samples

    def void update(self,
            SIZE_t* samples_win,
            SplitRecord split_record, 
            double* Xf  
            ) nogil:       
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

            for k in range(data.n_outputs):
                    
                class_label =<SIZE_t> data.y[ s_i * data.y_stride + k] # y[i,k]
                    
                if feature.type == FEATURE_CONTINUOUS:
                    if Xf[p] < split_record.threshold:
                        f_i = 0
                    else:
                        f_i = 1
                else:
                    f_i = Xf[p]

                label_index = f_i * feature_stride            # label_count[ f_i, k, label ]
                            + k   * label_count_stride
                            + class_label

                label_count[label_index] += w

    # gini
    def double node_impurity(self, double* label_count, SIZE_t wn_samples, double epsilon):

        cdef double* label_count = label_count 
        cdef SIZE_t wn_samples = wn_samples + noise(epsilon, self.random_state) # XXX

        cdef double total, gini, count
        cdef SIZE_t k, c
        cdef SIZE_t n_outputs = self.data.n_outputs 
        cdef SIZE_t* n_classes = self.data.n_classes 
        cdef SIZE_t label_count_stride = self.label_count_stride

        total = 0.0
        for k in range(n_outputs):
            gini = 0.0

            for c in range(n_classes[k]):
                count = label_count[c] + noise(epsilon, self.random_state) # XXX
                gini += (count*count) / wn_samples 
            
            total += gini
            label_count += label_count_stride 

        return total / n_outputs 


    def double improvement(self, SIZE_t* wn_subnodes_samples, SIZE_t n_subnodes, double epsilon):
        '''
        calculate improvement based on class distribution
        '''
        cdef double* label_count = self.label_count
       
        cdef double improvements = 0

        cdef SIZE_t i = 0
        # sum up node_impurity of each subset 
        for i in range(n_subnodes):
            label_count += self.feature_stride 
            improvement += self.node_impurity(label_count, wn_subnodes_samples[i], epsilon) # XXX

        return improvement

    def void node_value(self, double* dist):
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

    def __cinit__(self,
                    Criterion criterion,
                    SIZE_t max_candid_features,
                    RandomState random_state):
        self.criterion = criterion
        self.data = None       
        self.random_state = random_state 
        
        self.samples_win = NULL
        self.start = 0
        self.end   = 0

        self.features_win = NULL
        self.n_features = 0

        self.feature_values  = NULL # tempory array
        self.max_candid_features = max_candid_features
    
    def __dealloc__(self):
        free(self.samples_win)
        free(self.features_win)
        free(self.feature_values)

    cdef void init(self, 
            Data data):
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
        cdef SIZE_t* features_win 
                = safe_realloc (&self.features_win, n_features)
        for i in range(n_features):
            features_win[i] = i

        safe_realloc(&self.feature_values, n_samples)

        # set data
        self.data = data

    cdef void node_reset(self, SIZE_t start, SIZE_t end, 
                    double* weighted_n_node_samples) nogil:
        ''' call once for each node 
            set start, end, 
            return weighted_n_node_samples'''        

        self.start = start
        self.end   = end

        self.criterion.init(
                            self.data,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

    cdef void _choose_split_point(self):
        pass

    cdef void _choose_split_feature(self, SplitRecord* split_record, 
                            SIZE_t n_node_features, double epsilon):
        pass

    cdef void node_split(self, 
            SplitRecord* split_record, 
            SIZE_t* n_node_features,
            double epsilon):
        ''' 
            Calculate:
                best feature for split,
                best split point (for continuous feature)
        '''
        # create and init split records
        cdef SplitRecord* feature_records 
                = <SplitRecord*> calloc(n_node_features[0], sizeof(SplitRecord))
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
        cdef SIZE_t f_i = n_node_features[0]-1
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

                # 
                self.criterion.update(samples_win, current, Xf)
                current.improvement = self.criterion.improvement(current.wn_subnodes_samples, current.n_subnodes)

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

    cdef void node_value(self,double* dest) nogil:
        self.criterion.node_value(dest)

cdef class LapSplitter(Splitter):

    cdef void _choose_split_point(self):
        raise("Laplace mech doesn't support continuous feature") 
        
    # choose the best
    cdef void _choose_split_feature(self,
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

    cdef void _choose_split_point(self):
        pass
    
    cdef void _choose_split_feature(self,
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
                self.random_state)
        
        best[0] = records[index]

cdef class NBTreeBuilder:

    cdef __cinit__(int diffprivacy_mech,
                    double budget,
                    Splitter splitter,
                    SIZE_t max_depth,
                    SIZE_t max_candid_features,
                    RandomState random_state
                    ):

        self.diffprivacy_mech = diffprivacy_mech
        self.budget = budget

        self.splitter = splitter

        self.max_depth = max_depth  # verified by Classifier
        self.max_candid_features = max_candid_features # verified

        self.tree = NULL    # XXX ??? pass parameter from python
        self.data = NULL
        
        self.random_state = random_state

    cpdef build(self,
                Tree    tree,
                Data    data):
        
        cdef Splitter splitter = self.splitter

        cdef Data data = data
        self.data = data
        
        cdef Tree tree = tree
        _init_tree(tree, max_depth) # init tree capacity based on max_depth
        self.tree = tree
        
        # set parameter for building tree 
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t max_candid_features = self.max_candid_features

        # set parameter for diffprivacy
        cdef DOUBLE_t budget = self.budget
        cdef SIZE_t diffprivacy_mech = self.diffprivacy_mech
        cdef double epsilon_per_depth = 0.0
        cdef double epsilon_per_action= 0.0   
        if diffprivacy_mech == LAP_DIFF_PRIVACY_MECH:
            epsilon_per_depth = budget/(max_depth+1)
            epsilon_per_action = epsilon_per_depth/2.0
        elif diffprivacy_mech ==  EXP_DIFF_RPIVACY_MECH:
            epsilon_per_action = 
                budget/( (2+data.n_continuous_features)*max_depth + 2)
        else:
            epsilon_per_action = NO_DIFF_PRIVACY_BUDGET
        
        printf("espilon per action is %d", epsilon_per_action)

        # ===================================================
        # recursively depth first build tree 

        splitter.init(data) # set samples_win, features_win
        SplitRecord split_record 

        cdef SIZE_t max_depth_seen = -1 # record the max depth ever seen
        cdef SIZE_t start, end, start_i, end_i,
                    depth,
                    parent, 
                    index,
                    n_node_features
        cdef SIZE_t n_node_samples
        cdef DOUBLE_t noisy_n_node_samples
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
                noisy_n_node_samples = double(n_node_samples) + noise(epsilon_per_action, self.random_state) # XXX
                # reset class distribution based on this node
                # node_reset
                #       fill label_total_count,
                #       calculate weighted_n_node_samples
                splitter.node_reset(start, end, &weighted_n_node_samples)


                # if leaf
                if depth >= max_depth 
                    or n_node_features <= 0
                    or noisy_n_node_samples/(data.max_n_feature_values*data.max_n_classes) < np.sqrt(2.0)/epsilon_per_action: # XXX
                    
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
                    noise_distribution(epsilon_per_action, tree.nodes[node_id].values, self.data, self.random_state)
                else:       
                    # inner node
                    # choose split feature
                    # XXX: refine node_split
                    splitter.node_split( &split_record, &n_node_features, epsilon_per_action )
                
                    node_id = tree._add_node(
                            parent,
                            index,
                            False,     # not leaf node
                            split_record.feature_index,
                            split_record.threshold,
                            split_record.n_values,  # number of children
                            n_node_samples, 
                            weighted_n_node_samples # XXX
                            )

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

# =========================================================================
# Utils
# ========================================================================

cdef double laplace(double b, RandomState rand) except -1 with gil:
    # No diffprivacy
    if b <= 0.0:
        return 0.0

    cdef double uniform = rand.rand()-0.5
    if uniform > 0.0:
        return -epsilon*ln(1.0-2*uniform)
    else:
        return +epsilon*ln(1.0+2*uniform) 

cdef double noise(double epsilon, RandomState rand):
    if epsilon == NO_DIFF_PRIVACY_BUDGET:
        return 0.0

    return laplace(1.0/epsilon, rand)

cdef void noise_distribution(double epsilon, double* dist, Data data, RandomState rand):

    cdef SIZE_t k, i
    
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
            dest[rand_int(n_classes[k],random_state)] = 1.0

        dest += stride


cdef int draw_from_distribution(double* dist, int size, RandomState rand) nogil except -1 :
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
        return rand.randint(n)

    dist[i] = dist[i]/total

    point = rand.rand()

    i = 0
    for i in range(size):
        current += dist[i]
        if current > point:
            return i

    return n-1


cdef int draw_from_exponential_mech(double epsilon, SplitRecord* records, int size, double sensitivity, RandomState rand):

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
            improvements[i] = exp(improvements[i]*epsilon/(2*sensitivity))

    ret_idx = draw_from_distribution(improvements, n_split_points, rand)

    free(improvements)

    return ret_idx


