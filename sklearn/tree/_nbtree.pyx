cdef struct Feature:

    cdef int type       # continuous or discrete feature
    
    cdef int n_values   # the number of distinct value of the feature
                        # for continuous feature, = 2  

    cdef double max     # for continuous feature only
    cdef double min     # for continuous feature only

cdef struct SplitRecord:
    SIZE_t feature      # the index of the feature to split
    SIZE_t threshold    # for continuous feature only
    double improvement  # the improvement by selecting this feature

cdef struct StackRecord:
    SIZE_t start
    SIZE_t end
    SIZE_t depth
    SIZE_t parent
    SIZE_t index        # the index of this node in parent's children array
    SIZE_t n_candid_features

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
cdef class Criterion:

    SIZE_t start
    SIZE_t end
    SIZE_t pos

    Feature feature 

    double* label_count_total
    double* label_count 

    def __cinit__(self, Data data):
        self.data = data

        self.samples = NULL

        self.start = 0
        self.end   = 0

        self.label_count_stride = data.max_labels 

        cdef SIZE_t n_elements = n_outputs * label_count_stride
        self.label_count_total = <double*> calloc(n_elements, sizeof(double))
        self.label_count = <double*> calloc(n_elements*data.max_feature_values)

        self.fvalue_stride = n_elements 

        if self.label_count_total == NULL 
        or self.label_count == NULL:
            raise MemoryError()

    def node_init(self, 
                   SIZE_t*   samples, 
                   SIZE_t    start, 
                   SIZE_t    end) nogil:
        '''
        For one node, called once,
        update class distribution in this node
            
        init: 
            label_count_total, 2d array ( n_outputs, max_n_classes )     
        '''
          
        cdef UINT32_t* random_state = &self.rand_r_state
        
        # Initialize fields
        self.sample_weight = sample_weight
        self.samples  = samples
        
        self.start    = start
        self.end      = end
        self.n_node_samples = end - start
        
        # Initialize label_count_total and weighted_n_node_samples
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
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
            SplitRecord split, 
            double* Xf
            ):       # only for continuous feature
        '''
        udpate:
            label_count, 3d array (n_values of current feature, n_outputs, max_n_classes)
        '''
        cdef Feature feature = self.data.features[split.feature_index]
        
        cdef SIZE_t start = self.start
        cdef SIZE_t end   = self.end

        if feature.type == FEATURE_CONTINUOUS:
            pos = split.pos
            

        else:
            #feature_value = 0  # feature value, from 0 to feature.n_values-1   
            for p in range(start, end):
                i = samples[p]
           
                if self.data.sample_weight != NULL:
                    w = self.data.sample_weight[i]
                else:
                    w = 1.0

                for k in range(n_outputs):
                    
                    label  = <SIZE_t> data.y[ i * data.y_stride + k]# y[i,k]
                    fvalue = Xf[p]

                    label_index = fvalue * fvalue_stride            # label_count[ fvalue, k, label ]
                                + k      * label_count_stride
                                + label

                    label_count[label_index] += w

    def double node_impurity(self, double* label_count, SIZE_t wn_samples):

        cdef double* label_count = label_count 
        cdef SIZE_t wn_samples = wn_samples 

        cdef double total, gini, count
        cdef SIZE_t k, c
        cdef SIZE_t n_outputs = self.data.n_outputs 
        cdef SIZE_t* n_classes = self.data.n_classes 
        cdef SIZE_t label_count_stride = self.label_count_stride

        total = 0.0
        for k in range(n_outputs):
            gini = 0.0

            for c in range(n_classes[k]):
                count = label_count[c]
                gini += (count*count) / wn_samples 
            
            total += gini
            label_count += label_count_stride 

        return total / n_outputs 


    def double improvement(self, SIZE_t* wn_subnodes_samples, SIZE_t n_subnodes):
        '''
        calculate improvement based on class distribution
        '''
        cdef double* label_count = self.label_count
       
        cdef improvements = 0

        # sum up node_impurity of each subset 
        for i in range(n_subnodes):
            label_count += self.fvalue_stride 
            improvement += self.node_impurity(label_count, wn_subnodes_samples[i])

        return improvement

    def node_value(self, double* dist):
        '''
        return class distribution of node, i.e. label_count_total
        '''
        pass 

cdef class Splitter:

    def __cinit__(self,
                    Criterion criterion,
                    SIZE_t max_candid_features,
                    object random_state):

        self.criterion = criterion
        self.samples = NULL
        self.features_window = NULL
        self.feature_values = NULL # tempory array
        self.max_candid_features = max_candid_features
        
        self.random = ... 
    
    cdef init(self, 
            Data data):
            #np.ndarray[DTYPE_t, ndim=2] X,
            #np.ndarray[double,  ndim=2] y,
            #Feature* features):
        '''
        set samples
        set features_window 
        set feature_values
        '''
        # set samples window
        cdef SIZE_t n_samples = data.n_samples
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i
        for i in range(n_samples):
            samples[i] = i

        # set features window
        cdef SIZE_t  n_features = data.n_features
        cdef SIZE_t* features_window 
                = safe_realloc (&self.features_window, n_features)
        for i in range(n_features):
            features_window[i] = i

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
            SIZE_t n_candid_features):
        ''' Find best feature for split,
            For continuous feature, also find best split point '''

        Data data = self.data

        SplitRecord* feature_records 
            = <SplitRecord*> calloc(n_candid_features, sizeof(SplitRecord))
        _init_split(feature_records, n_candid_features)

        SplitRecord current
        SplitRecord best

        SIZE_t* features_window = self.features_window 

        SIZE_t f_i = n_candid_features-1
        SIZE_t f_j = 0

        DOUBLE_t* Xf = self.feature_values

        while f_j <= f_i :
            current.feature_index = features_window[f_j]

            # copy to Xf
            for p in range(start, end):
                Xf[p] = data.X [    # X[sample_index, feature_index] 
                                data.X_sample_stride  * samples[p] + 
                                data.X_feature_stride * current.feature_index
                                ]
            
            sort(Xf+start, samples+start, end-start)

            # if constant feature
            if Xf[end-1] <= Xf[start] + FEATURE_THRESHOLD:
                features_window[f_j] = features_window[f_i]
                features_window[f_i] = current.feature
                f_i --

            # not constant feature
            else:                
                # if continuous feature
                if data.features[ current.feature_index ].type == FEATURE_CONTINUOUS:
                    _choose_split_point(&current)
                
                else:
                    current.n_values = self.data.features[current.feature].n_values

                    n_subnodes_samples  = <SIZE_t*> safe_realloc(current.n_subnodes_samples, feature.n_values)
                    wn_subnodes_samples = <SIZE_t*> safe_realloc(current.wn_subnodes_samples, feature.n_values)
                    
                    for i in range(start,end):
                        w = sample_weight[ i ] 
                        n_subnodes_samples[ Xf[i] ] ++
                        wn_subnodes_samples[ Xf[i] ] += w

                self.criterion.update(current)
                current.improvement = self.criterion.improvement(current.wn_subnodes_samples, current.n_values)

                if current.improvement > best.improvement
                    best = current

                f_j ++

        
        _choose_split_feature(feature_records, &best )

        split_record[0] = best
        n_candid_features[0] = f_i+1 

cdef class LapSplitter(Splitter):

    cdef _choose_split_point():

    cdef _choose_split_feature():

cdef class ExpSplitter(Splitter):

    cdef _choose_split_point():

    cdef _choose_split_feature():



cdef class DepthFirstNBTreeBuilder:

    cdef __cinit__(int diffprivacy_mech,
                    double budget,
                    Splitter splitter,
                    max_depth,
                    max_candid_features):

    cpdef build(self,
                Tree    tree,
                np.ndarray  X,
                np.ndarray  y,
                np.ndarray  sample_weight=None,
                Feature* features):


        self._features = features 

        # parameters setting in __cinit__
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t max_candid_features = self.max_candid_features

        # calculate epsilon based on diffprivcy mech and max depth
        cdef double epsilon_per_action
        printf("espilon per action is %d", epsilon_per_action)

        # XXX: splitter init
        splitter.init(X, y, features)
        #######################
        # recursively partition 
        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        # root node record
        stack_record.start = 0
        stack_record.end   = n_node_samples 
        stack_record.depth = 0
        stack_record.parent=_TREE_UNDEFINED
        stack_record.index = 0
        stack_record.n_candid_features  = 0
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
                n_candid_features = stack_record.n_candid_features 

                n_node_samples = end - start
                noise(n_node_samples, epsilon_per_action)

                # reset class distribution based on this node
                # XXX: node_reset
                splitter.node_reset(start, end, &weighted_n_node_samples)

                # if leaf
                if depth >= max_depth 
                    or len(candid_features) <= 0
                    or n_node_samples ... :
                    
                    # leaf node
                    node_id = tree._add_node(parent,
                                             index,
                                             True,      # leaf node
                                             NO_FEATURE,
                                             NO_THRESHOLD,
                                             n_node_samples,
                                             weighted_n_node_samples
                                             )
                
                    # store class distribution into node.values
                    # XXX: node_value
                    splitter.node_value(tree, node_id)
                    # add noise to the class distribution
                    noise_distribution(tree.node[node_id].values, epsilon_per_action)
                else:       
                    # inner node
                    # choose split feature
                    # XXX: node_split
                    splitter.node_split( &split_record )
                 
                    node_id = tree._add_node(parent,
                                             index,
                                             False,     # not leaf node
                                             split_record.feature,
                                             split_record.threshold,
                                             n_node_samples,
                                             weighted_n_node_samples
                                             )

                    # push children into stack
                    split_feature = self._feature[split_record.feature]   
                    for i in range(split_feature.n_values):
                        rc = stack.push(
                            split_record.starts[i], # start pos
                            split_record.ends[i],   # end pos
                            depth+1,                # depth of this new node
                            node_id,                # child's parent id
                            i,                      # the index
                            n_candid_features 
                            )
                    
                        if rc == -1:
                            raise MemoryError()
                            
                if depth > max_depth_seen:
                    max_depth_seen = depth


            # TODO: prune


cdef class Tree:

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


