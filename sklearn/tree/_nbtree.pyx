from libc.stdio cimport printf
from libc.stdlib cimport calloc, free, realloc, exit
from libc.string cimport memcpy, memset
from libc.math   cimport log, exp, sqrt, pow
from cpython cimport Py_INCREF, PyObject

from sklearn.tree._nbutils cimport Stack, StackRecord

from sklearn.tree._tree cimport sort, rand_int, rand_double # XXX

import numpy as np
cimport numpy as np
np.import_array() # XXX

from scipy.stats import norm

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
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).children,  # XXX
        <Py_ssize_t> &(<Node*> NULL).n_children,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples
    ]
})


cpdef public SIZE_t NO_DIFF_PRIVACY_MECH  = 0
cpdef public SIZE_t LAP_DIFF_PRIVACY_MECH = 1
cpdef public SIZE_t EXP_DIFF_RPIVACY_MECH = 2

cpdef public DOUBLE_t NO_DIFF_PRIVACY_BUDGET = -1.0

cpdef SIZE_t NO_THRESHOLD = -1
cpdef SIZE_t NO_FEATURE = -1
cpdef SIZE_t FEATURE_CONTINUOUS = 0
cpdef SIZE_t FEATURE_DISCRETE   = 1

# ====================================================================
# Criterion
# ====================================================================

cdef class Criterion:

    def __cinit__(self, DataObject dataobject, object random_state, bint debug):
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

        self.debug = debug
       
        cdef Data* data = dataobject.data

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

        # self.sensitivity = 2.0   # gini
    
        self.data = data

    def __dealloc__(self):
        free(self.label_count)
        free(self.label_count_total)

    cdef void init(self, 
                   Data*      data,
                   SIZE_t*   samples_win, 
                   SIZE_t    start, 
                   SIZE_t    end,
                   ): # nogil:
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
            SplitRecord* split_record, 
            DTYPE_t* Xf  
            ): # nogil:       
        '''
        udpate:
            label_count, array[n_subnodes][n_outputs][max_n_classes]
        '''
        cdef Feature* feature = &self.data.features[split_record.feature_index]
       
        cdef Data* data = self.data  # XXX
        cdef SIZE_t start = self.start
        cdef SIZE_t end   = self.end

        cdef DOUBLE_t* label_count = self.label_count 
        cdef SIZE_t feature_stride = self.feature_stride 
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef SIZE_t class_label
        cdef SIZE_t label_index
        cdef SIZE_t p, s_i, f_i, k, 
        cdef DOUBLE_t w

        cdef SIZE_t debug = 0
        if debug:
            printf("Criterion_update(): Begin to update label_count samples [%d, %d]\n", start, end)

        memset(label_count, 0, feature.n_values*feature_stride*sizeof(DOUBLE_t))

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
                        #if debug:
                        #    printf("%.2f, ", Xf[p])
                else:
                    f_i = <SIZE_t>Xf[p] # XXX type conversion
        
                # label_count[ f_i, k, label ]
                label_index = f_i*feature_stride + k*label_count_stride + class_label
                
                if label_index > feature_stride*feature.n_values:
                    printf("data.y, %f\n", data.y[s_i*data.y_stride])
                    printf("data.y, %u\n", <SIZE_t>data.y[s_i*data.y_stride])
                   
                    printf("from %u to %u\n", start, end)
                    for p in range(0, data.n_samples):
                        printf("%f, ",data.y[p*data.y_stride])
                    printf("\n") 
                    
                    printf("f[%u]=%u, f_stride[%u], class[%u]\n",split_record.feature_index, f_i, feature_stride, class_label)
                    printf("label index is %u\n", label_index)
                    exit(1) 
                
                label_count[label_index] += w

        if debug:
            #printf("\n")
            label_count = self.label_count
            for f_i in range(feature.n_values):
                printf("Fvalue[%u]: ", f_i)
                for k in range(data.n_outputs):
                    for s_i in range(data.max_n_classes):
                        printf("%.0f, ", label_count[f_i*feature_stride + k*label_count_stride + s_i])
                printf("\n")         

    cdef DOUBLE_t node_impurity(self): # nogil:
        pass 

    cdef DOUBLE_t children_impurity(self, DOUBLE_t* label_count, DOUBLE_t wn_samples, DOUBLE_t epsilon): # nogil:
        pass

    cdef DOUBLE_t improvement(self, DOUBLE_t* wn_subnodes_samples, SIZE_t n_subnodes,
                            DOUBLE_t impurity, DOUBLE_t epsilon) : #nogil:
        '''calculate improvement based on class distribution'''
        cdef DOUBLE_t* label_count = self.label_count
        cdef DOUBLE_t improvement = 0.0
        
        cdef SIZE_t debug = 0
        
        if debug:
            printf("improvement(): begin to calculate improvement:\n")

        # sum up children_impurity of each subset 
        cdef SIZE_t i = 0
        cdef DOUBLE_t tmp = 0.0
        for i in range(n_subnodes):
            tmp = self.children_impurity(label_count, wn_subnodes_samples[i], epsilon)
           
            # the standard version
            #tmp = (wn_subnodes_samples[i]/self.weighted_n_node_samples)*tmp

            # KDD10 paper version 
            tmp = wn_subnodes_samples[i]*tmp

            improvement += tmp 
            if debug:
                printf("subnode[%d] improvement = %f\n", i, tmp)
            label_count += self.feature_stride 
        
        if debug:
            printf("impurity-improvement: %f-%f=%f\n",impurity, improvement, impurity-improvement)

        # the standard version
        #improvement = impurity - improvement
        
        # KDD10 version
        if improvement < 0.0:
            printf("improvement(): Error, improvement %f is less than 0\n",improvement) 
            printf("%f-%f=%f\n", impurity, improvement, impurity-improvement)
            exit(1)
      
        # KDD10 paper version
        improvement = -improvement
        return improvement


    cdef void node_value(self, DOUBLE_t* dest): # nogil:
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

            if 0:
                for c in range(n_classes[k]):
                    printf("%.0f ",label_count_total[c])

            dest += label_count_stride
            label_count_total += label_count_stride 

    cdef void print_distribution(self, DOUBLE_t* dist): # nogil:
        
        cdef SIZE_t n_outputs = self.data.n_outputs
        cdef SIZE_t* n_classes = self.data.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef DOUBLE_t* label_count_total = dist 

        if label_count_total == NULL:
            #printf("Warning, dist is NULL\n")
            label_count_total = self.label_count_total

        cdef SIZE_t k
        for k in range(n_outputs):
            for c in range(n_classes[k]):
                printf("%.2f ",label_count_total[c])
            
            label_count_total += label_count_stride 
    
cdef class Gini(Criterion):

    def __cinit__(self, DataObject dataobject, object random_state, bint debug):
        
        self.sensitivity = 2.0 

    cdef DOUBLE_t node_impurity(self): # nogil:
        cdef DOUBLE_t total, gini, count, tmp
        cdef SIZE_t k, c
        
        cdef SIZE_t n_outputs = self.data.n_outputs 
        cdef SIZE_t* n_classes = self.data.n_classes 
        cdef SIZE_t label_count_stride = self.label_count_stride
        
        cdef DOUBLE_t* label_count_total = self.label_count_total
        cdef DOUBLE_t weighted_n_node_samples = self.weighted_n_node_samples

        cdef SIZE_t debug = 0
        if debug:
            printf("label_count_total:")
            for k in range(n_outputs):
                for c in range(n_classes[k]):
                    printf("%.0f, ", label_count_total[k*label_count_stride + c])
            printf("\n")

        if weighted_n_node_samples <= 0.0:
            if debug:
                printf("wn_samples <=0, cancel to count this child's impurity")
            return 0.0

        total = 0.0
        for k in range(n_outputs):
            gini = 0.0
            for c in range(n_classes[k]):
                count = label_count_total[c]
                
                if count < 0.0:
                    printf("Error label_count[%d]=%d should be greater than 0.0", c, count)
                    exit(0)
                if count == 0.0:
                    if debug:
                        printf("label_count=0.0, cancel to calculate this part\n")
                    continue

                tmp = count/weighted_n_node_samples
                gini += (1.0 - tmp*tmp )
                
                if debug:    
                    printf("gini[%f]=1.0-(%f/%f)^2\n", gini, count, weighted_n_node_samples)

            total += gini
            label_count_total += label_count_stride 
        
        if total < 0.0:
            printf("Error node impurity(Gini) %f should be greater than 0.0\n", total)
            exit(0)

        if debug:
            printf("node_impurity(Gini)[%.2f]\n", total/n_outputs)
        return total/n_outputs 

    # gini
    cdef DOUBLE_t children_impurity(self, DOUBLE_t* label_count, DOUBLE_t wn_samples, DOUBLE_t epsilon): # nogil:

        cdef DOUBLE_t total, gini, count, tmp
        cdef SIZE_t k, c

        cdef SIZE_t n_outputs = self.data.n_outputs 
        cdef SIZE_t* n_classes = self.data.n_classes 
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef UINT32_t* rand = &self.rand_r_state

        cdef SIZE_t debug = 0
        if debug:
            printf("label_count:\n")
            for k in range(n_outputs):
                for c in range(n_classes[k]):
                    printf("%f, ", label_count[k*label_count_stride + c])
            printf("\n")


        if wn_samples <= 0.0:
            if debug:
                printf("wn_samples <=0, cancel to count this child's impurity")
            return 0.0

        total = 0.0
        for k in range(n_outputs):
            gini = 0.0
            for c in range(n_classes[k]):
                count = label_count[c] 
                
                if count < 0.0:
                    printf("Error label_count[%d]=%d should be greater than 0.0", c, count)
                    exit(0)
                if count == 0.0:
                    if debug:
                        printf("label_count=0.0, cancel to calculate this part\n")
                    continue

                tmp = count/wn_samples 
                gini += (1.0 - tmp*tmp )

                if debug:    
                    printf("gini[%f]=1.0-(%f/%f)^2\n", gini, count, wn_samples)

            total += gini
            label_count += label_count_stride 
        
        if total < 0.0:
            printf("Error children impurity(Gini) %f should be greater than 0.0\n", total)
            exit(0)

        if debug:
            printf("children_impurity(Gini)[%f]\n", total/n_outputs)
        return total/n_outputs 

cdef class Entropy(Criterion):
    
    def __cinit__(self, DataObject dataobject, object random_state, bint debug):
        
        cdef Data* data = dataobject.data
        self.sensitivity = log(data.n_samples)/log(2.0)+1  # sensitivity = log2(n_samples)+1

    cdef DOUBLE_t node_impurity(self): # nogil:
   
        cdef DOUBLE_t total, entropy, count, tmp
        cdef SIZE_t k, c
        
        cdef SIZE_t n_outputs = self.data.n_outputs 
        cdef SIZE_t* n_classes = self.data.n_classes 
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef DOUBLE_t* label_count_total = self.label_count_total

        cdef DOUBLE_t weighted_n_node_samples = self.weighted_n_node_samples

        cdef SIZE_t debug = 0
        if debug:
            printf("label_count_total:")
            for k in range(n_outputs):
                for c in range(n_classes[k]):
                    printf("%.0f, ", label_count_total[k*label_count_stride + c])
            printf("\n")

        if weighted_n_node_samples <= 0.0:
            if debug:
                printf("entropy node_impurity: wn_samples %f <= 0.0, return 0.0\n", weighted_n_node_samples)
            return 0.0

        total = 0.0
        for k in range(n_outputs):
            entropy = 0.0
            for c in range(n_classes[k]):
            
                count = label_count_total[c]
                
                if count < 0.0:
                    printf("Error label_count[%d]=%f should be greater than 0.0", c, count)
                    exit(0)
                if count == 0.0:
                    if debug:
                        printf("label_count=0.0, cancel to calculate this part\n")
                    continue

                tmp = count/weighted_n_node_samples 
                entropy -= tmp*log(tmp)
                
                if debug:    
                    printf("(%.0f/%.2f)log(%.0f/%.2f) = %.2f\n", 
                            count, weighted_n_node_samples, 
                            count, weighted_n_node_samples, tmp)

            total += entropy 
            label_count_total += label_count_stride 
        
        if total < 0.0:
            printf("Error node impurity %f should be greater than 0.0", total)
            exit(0)

        if debug:
            printf("entropy node_impurity[%.2f]\n", total/n_outputs)
        return total/n_outputs 

    # entropy
    cdef DOUBLE_t children_impurity(self, DOUBLE_t* label_count, DOUBLE_t wn_samples, DOUBLE_t epsilon): # nogil:

        cdef DOUBLE_t total, entropy , count, tmp
        cdef SIZE_t k, c

        cdef SIZE_t n_outputs = self.data.n_outputs 
        cdef SIZE_t* n_classes = self.data.n_classes 
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef UINT32_t* rand = &self.rand_r_state
        
        cdef SIZE_t debug = 0

        if wn_samples <= 0.0:
            if debug:
                printf("wn_samples <= 0, cancel to calculate this child's impurity\n")
            return 0.0

        if debug:
            printf("label_count:\n")
            for k in range(n_outputs):
                for c in range(n_classes[k]):
                    printf("%f, ", label_count[k*label_count_stride + c])
            printf("\n")

        total = 0.0
        for k in range(n_outputs):
            entropy  = 0.0
            for c in range(n_classes[k]):
                
                count = label_count[c] 

                if count < 0.0:
                    printf("Error label_count[%d]=%f should be greater than 0.0", c, count)
                    exit(0)
                if count == 0.0:
                    if debug:
                        printf("label_count=0.0, cancel to calculate this part\n")
                    continue

                tmp =  count/wn_samples
                entropy -= tmp*log(tmp)
              
                if debug:    
                    printf("(%.0f/%.2f)log(%.0f/%.2f) = %.2f\n", 
                            count, wn_samples, count, wn_samples, tmp)

            total += entropy 
            label_count += label_count_stride 

        if total < 0.0:
            printf("Error children impurity %d should be greater than 0.0", total)
            exit(0)

        if debug:
            printf("entorpy children_impurity[%f]\n", total/n_outputs)
        return total/n_outputs 


cdef class LapEntropy(Entropy):
   
    # laplace entropy
    cdef DOUBLE_t children_impurity(self, DOUBLE_t* label_count, DOUBLE_t wn_samples, DOUBLE_t epsilon): # nogil:

        cdef bint debug = 0
        if debug:
            printf("get into LapEntropy's children_impurity, wn_samples is %f, epsilon is %f\n", wn_samples, epsilon)
        cdef UINT32_t* rand = &self.rand_r_state
        cdef DOUBLE_t total, entropy , count, tmp
        cdef SIZE_t k, c
        cdef SIZE_t n_outputs = self.data.n_outputs 
        cdef SIZE_t* n_classes = self.data.n_classes 
        cdef SIZE_t label_count_stride = self.label_count_stride

        wn_samples += noise(epsilon, rand) 
        if wn_samples <= 0.0:
            return 0.0

        total = 0.0
        for k in range(n_outputs):
            entropy  = 0.0
            for c in range(n_classes[k]):
                if debug:
                    printf("label_count[%u] is %f\n", c, label_count[c])

                count = label_count[c] + noise(epsilon, rand) # XXX what if count < 0.0
                if debug:
                    printf("after noise is %f\n", count)
                if count <= 0.0:
                    continue
                
                tmp   = count/wn_samples
                entropy -= count*log(tmp)
            total += entropy 
           
            if debug:
                printf("entropy %f\n", entropy)

            label_count += label_count_stride 
        
        return total/n_outputs 

    # Laplace Mech Entropy
    cdef DOUBLE_t improvement(self, DOUBLE_t* wn_subnodes_samples, SIZE_t n_subnodes,
                            DOUBLE_t impurity, DOUBLE_t epsilon) : #nogil:
        ''' calculate improvement based on class distribution'''
        cdef DOUBLE_t* label_count = self.label_count
        cdef DOUBLE_t improvement = 0.0
        cdef DOUBLE_t tmp = 0.0
        cdef DOUBLE_t epsilon_per_action = epsilon/2.0  # one for noise(wn_subnodes_samples) 
                                                        # one for noise(count) 
   
        cdef bint debug = 1
        if debug:
            if label_count == NULL:
                printf("Error: label_count is Null\n")
            printf("get into improvemnt()\n")

        # sum up children_impurity of each subset 
        cdef SIZE_t i = 0
        for i in range(n_subnodes):
            tmp = self.children_impurity(label_count, wn_subnodes_samples[i], epsilon_per_action)
            # tmp = tmp/self.weighted_n_node_samples
            improvement += tmp 
           
            if debug:
                printf("subnodes impv is %f, wn_samples is %f\n", tmp, self.weighted_n_node_samples)

            label_count += self.feature_stride 
       
        if 0 and improvement < 0.0:
            printf("Warning, improvement %f is less than 0\n",improvement) 
        
        if debug:
            printf("improvement is %f\n", improvement)

        return improvement


# ===========================================================
# Splitter 
# ===========================================================
cdef inline void _init_split(SplitRecord* self): #nogil:
    self.feature_index = -1
    self.threshold = 0.0
    self.improvement = -INFINITY

    self.n_subnodes = 0
    self.n_subnodes_samples = NULL
    self.wn_subnodes_samples = NULL

cdef inline void _copy_split(SplitRecord* current, SplitRecord* best):
    best.feature_index = current.feature_index
    best.threshold = current.threshold
    best.improvement = current.improvement
    best.n_subnodes  = current.n_subnodes
    best.n_subnodes_samples = NULL
    best.wn_subnodes_samples = NULL
    best.n_subnodes_samples = <SIZE_t*>calloc(best.n_subnodes, sizeof(SIZE_t))
    best.wn_subnodes_samples= <DOUBLE_t*>calloc(best.n_subnodes, sizeof(SIZE_t))
    #safe_realloc(&best.n_subnodes_samples, best.n_subnodes)
    #safe_realloc(&best.wn_subnodes_samples, best.n_subnodes)
    for i in range(best.n_subnodes):
        best.n_subnodes_samples[i] = current.n_subnodes_samples[i]
        best.wn_subnodes_samples[i]= current.wn_subnodes_samples[i]

    if 0:
        printf("best %p, current %p\n", best.n_subnodes_samples, current.n_subnodes_samples)

cdef class Splitter:


    def __cinit__(self,
                    Criterion criterion,
                    SIZE_t max_candid_features,
                    object random_state, 
                    bint debug):

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
   
        self.debug = debug
    
    def __dealloc__(self):
        free(self.samples_win)
        free(self.features_win)
        free(self.feature_values)

    cdef void init(self, 
            Data* data) except *:
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
            DOUBLE_t* weighted_n_node_samples): # nogil:
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

    cdef void _choose_split_point(self, SplitRecord* best, DTYPE_t* Xf, DOUBLE_t impurity, DOUBLE_t epsilon): 
                                                                                                        # nogil
        pass

    cdef SIZE_t _choose_split_feature(self, SplitRecord* split_records, SIZE_t size, DOUBLE_t epsilon): # nogil:
        pass

    cdef SplitRecord* node_split(self, 
            SIZE_t* ptr_n_node_features,
            DOUBLE_t impurity,
            SIZE_t diffprivacy_mech,
            DOUBLE_t epsilon) except *:
        ''' 
            Calculate:
                best feature for split,
                best split point (for continuous feature)
        '''
        # create and init split records
        cdef SIZE_t n_node_features = ptr_n_node_features[0]

        cdef SplitRecord* feature_records = <SplitRecord*> calloc(n_node_features, sizeof(SplitRecord))
        cdef SIZE_t i
        for i in range(n_node_features):
            _init_split(&feature_records[i])

        cdef SplitRecord* current = NULL
        cdef SplitRecord* best = <SplitRecord*> calloc(1, sizeof(SplitRecord))
        _init_split(best)
       
        cdef DOUBLE_t best_improvement = -INFINITY
        cdef SIZE_t best_i = -1

        cdef Data* data = self.data
        cdef SIZE_t* samples_win = self.samples_win
        cdef Feature* feature
        cdef SIZE_t* features_win = self.features_win 
        #cdef SIZE_t f_i = n_node_features
        cdef SIZE_t f_j = 0
        cdef SIZE_t tmp

        cdef DTYPE_t* Xf  = self.feature_values
        cdef SIZE_t start = self.start
        cdef SIZE_t end   = self.end
        cdef DOUBLE_t w   = 0.0

        cdef SIZE_t* n_subnodes_samples
        cdef DOUBLE_t* wn_subnodes_samples

        cdef bint debug = 1

        # only for laplace mech
        cdef DOUBLE_t epsilon_per_feature = epsilon / n_node_features
        if debug:
            printf("node_split(): epsilon_per_feature is %f\n", epsilon_per_feature)

        if debug:
            printf("node_split(): selecting feature from %u candids, samples %d to %d\n",
                    n_node_features, start, end)
            printf("node_split(): feature windows:")
            for f_j in range(n_node_features):
                printf("%u, ",features_win[f_j])
            printf("\n")
        
        f_j = 0
        while f_j < n_node_features :
            # copy to Xf
            for p in range(start, end):
                # Xf[p] = X[sample_index, feature_index]
                Xf[p] = data.X [     
                        samples_win[p]    * data.X_sample_stride  
                     +  features_win[f_j] * data.X_feature_stride ]
            
            # use _tree.py sort() 
            sort( Xf+start, samples_win+start, end-start)
    
            # if constant feature
            if Xf[end-1] <= Xf[start] + FEATURE_THRESHOLD:
                if debug:
                    printf("node_split(): f[%u] X[%d]=%f, X[%d]=%f is close, ignore this f\n",
                            features_win[f_j], start, Xf[start], end-1, Xf[end-1])

                tmp                             = features_win[f_j] 
                features_win[f_j]               = features_win[n_node_features-1]
                features_win[n_node_features-1] = tmp

                n_node_features -= 1
                # goes to next candidate feature

            # not constant feature.format(best_i, n_node_features)
            else:                
                current = &feature_records[f_j]
                current.feature_index = features_win[f_j]
                feature = &data.features[ features_win[f_j] ]

                # if continuous feature
                if feature.type == FEATURE_CONTINUOUS:
                    if debug:
                        printf("node_split(): continuous feature[%u],begin to choose split point\n", 
                                            current.feature_index )

                    if diffprivacy_mech == LAP_DIFF_PRIVACY_MECH:
                        printf("node_split(): Error, Laplace Mech does not support continuous feature\n")
                        exit(1)

                    current.n_subnodes = 2
                    self._choose_split_point(current, Xf, impurity, epsilon)

                    if debug:
                        printf("node_split(): continuous feature[%u]\n", features_win[f_j])
                        printf("                threshold %6.1f\n", current.threshold)
                        printf("                improvement %6.1f\n", current.improvement )
                        printf("                nodes:\n")
                        for i in range(current.n_subnodes):
                            printf("                subnode[%d] n=%4d, wn=%4.1f\n",
                                    i, current.n_subnodes_samples[i], 
                                    current.wn_subnodes_samples[i])
                
                else:
                    current.n_subnodes = feature.n_values
                   
                    current.n_subnodes_samples = <SIZE_t*>calloc(current.n_subnodes, sizeof(SIZE_t))
                    current.wn_subnodes_samples= <DOUBLE_t*>calloc(current.n_subnodes,sizeof(DOUBLE_t))
                    n_subnodes_samples = current.n_subnodes_samples
                    wn_subnodes_samples= current.wn_subnodes_samples
                    memset(n_subnodes_samples, 0, sizeof(SIZE_t)*current.n_subnodes)
                    memset(wn_subnodes_samples,0, sizeof(DOUBLE_t)*current.n_subnodes)

                    for i in range(start,end):
                        if data.sample_weight == NULL:
                            w = 1.0
                        else:
                            w = data.sample_weight[ i ]

                        n_subnodes_samples [ <SIZE_t>Xf[i] ] += 1 
                        wn_subnodes_samples[ <SIZE_t>Xf[i] ] += w

                    if debug:
                        printf("node_split(): feature[%u]:\n",features_win[f_j])
                        for i in range(current.n_subnodes):
                            printf("            subnodes[%d] n=%d, wn=%f\n",i, current.n_subnodes_samples[i], current.wn_subnodes_samples[i])
                        printf("\n")
                
                    self.criterion.update(samples_win, current, Xf)
                    current.improvement = self.criterion.improvement(
                                                current.wn_subnodes_samples,
                                                current.n_subnodes,
                                                impurity, 
                                                epsilon_per_feature)  # XXX only for laplace mech

                if debug:
                    printf("node_split(): [%d] f[%u] score=%.1f\n", f_j, current.feature_index, current.improvement)
                
                if current.improvement > best_improvement:
                    best_improvement = current.improvement
                    best_i = f_j
                f_j += 1

        if diffprivacy_mech == EXP_DIFF_RPIVACY_MECH:
            best_i = self._choose_split_feature(feature_records, 
                                                n_node_features,    # n_features
                                                epsilon)  # XXX only for exponential mech

        # if there's no any feature which can be splitted 
        if best_i < 0 or best_i >= n_node_features:
            if best_i == -1:
                if n_node_features != 0 or f_j != 0:
                    raise ValueError("Error, n_node_features[{0}] and f_j[{1}] should be 0 \
                                        when there is not best split feature".format(n_node_features, f_j))
                    exit(1)

                return NULL
                
            raise ValueError("best split feature index {0} \
                              should btw [0, {2})\n".format(best_i, n_node_features))
            exit(1)


        _copy_split(&feature_records[best_i], best)
       
        # switch best_i and n_node_features-1 in features_win
        tmp                  = features_win[best_i]
        features_win[best_i] = features_win[n_node_features-1]
        features_win[n_node_features-1]  = tmp

        if 0:
            printf("selected i[%d] f[%u] score %f\n", best_i, best.feature_index, best.improvement)
        if debug:
            if best.improvement == 0.0:
                printf("node impurity is %f\n", impurity)
                printf("distribution of feature[%u]:\n",best.feature_index)
                for i in range(best.n_subnodes):
                    printf("%u ",best.n_subnodes_samples[i])
                printf("\n")
 
        # free
        for i in range(n_node_features):
            free(feature_records[i].n_subnodes_samples)
            free(feature_records[i].wn_subnodes_samples)
        free(feature_records)
        
        ptr_n_node_features[0] = n_node_features-1
        return best

    cdef void node_value(self,DOUBLE_t* dest):# nogil:
        self.criterion.node_value(dest)
    
    cdef DOUBLE_t node_impurity(self): # nogil:
        return self.criterion.node_impurity()

cdef class LapSplitter(Splitter):

    cdef void _choose_split_point(self, SplitRecord* current, DTYPE_t* Xf, DOUBLE_t impurity, DOUBLE_t epsilon):# nogil:
        
        if 1:
        # with gil:
            raise("Laplace mech doesn't support continuous feature") 
        
    # choose the best
    cdef SIZE_t _choose_split_feature(self,
        #        SplitRecord* best, 
                SplitRecord* records,
                SIZE_t size,
                DOUBLE_t epsilon):# nogil:

        ''' Choose the best split feature ''' 
        cdef SIZE_t i, max_index = 0
        cdef DOUBLE_t max_improvement = -INFINITY

        for i in range(size):
            if records[i].improvement > max_improvement:
                max_index = i
                max_improvement = records[i].improvement 
        
        #best[0] = records[max_index]
        return max_index

cdef class ExpSplitter(Splitter):

    cdef void _choose_split_point(self, SplitRecord* best, DTYPE_t* Xf,DOUBLE_t impurity, DOUBLE_t epsilon):
                                                                                        # nogil:
        cdef SIZE_t best_i
        cdef DOUBLE_t best_improvement = -INFINITY
        cdef SIZE_t start = self.start
        cdef SIZE_t end   = self.end
        cdef SIZE_t n_split_points = end - start

        cdef SIZE_t* samples_win = self.samples_win
        cdef SIZE_t* n_subnodes_samples
        cdef DOUBLE_t* wn_subnodes_samples
        cdef SplitRecord* current
        cdef SplitRecord* records

        cdef UINT32_t* rand = &self.rand_r_state
        cdef DOUBLE_t sensitivity = self.criterion.sensitivity

        #cdef bint debug = self.debug
        cdef bint debug = 0
        if debug:
            printf("Choosing split points for feature[%d], samples %d to %d\n", best.feature_index, start, end)

        if n_split_points <=0 :
            return

        records = <SplitRecord*>calloc(n_split_points, sizeof(SplitRecord))

        cdef SIZE_t i
        for i in range(n_split_points):
            _init_split(&records[i])
            records[i].feature_index = best.feature_index
            records[i].n_subnodes = 2

        cdef double* weights
        weights = <double*>calloc(n_split_points, sizeof(double))
        # memset

        cdef SIZE_t start_p = start
        cdef SIZE_t end_p   = start
        cdef SIZE_t idx = 0
        while end_p < end:
            
            start_p = end_p
            
            # if Xf[p] and Xf[p+1] is of little difference, skip evaluating Xf[p]
            while (end_p + 1 < end and Xf[end_p + 1] <= Xf[end_p] + FEATURE_THRESHOLD):
                end_p += 1
            
            # (p + 1 >= end) or (x[samples[p + 1], current.feature] >
            #                    x[samples[p], current.feature])
            end_p += 1
            # (p >= end) or (x[samples[p], current.feature] >
            #                x[samples[p - 1], current.feature])
            
            if end_p < end:
                # threshold = [ Xf[start_p], Xf[end_p] )
                if Xf[start_p] == Xf[end_p]: 
                    printf("_choose_split_point(): Error, Xf[%d] == Xf[%d] = %f recursively occur\n", start_p, end_p, Xf[start_p])
                    exit(1)

                current = &records[idx] 
                current.n_subnodes = 2

                weights[idx] = Xf[end_p] - Xf[start_p] 
                 
                current.n_subnodes_samples = <SIZE_t*>calloc(current.n_subnodes, sizeof(SIZE_t))
                current.wn_subnodes_samples= <DOUBLE_t*>calloc(current.n_subnodes,sizeof(DOUBLE_t))
                n_subnodes_samples = current.n_subnodes_samples
                wn_subnodes_samples= current.wn_subnodes_samples
                memset(n_subnodes_samples, 0, sizeof(SIZE_t)*current.n_subnodes)
                memset(wn_subnodes_samples,0, sizeof(DOUBLE_t)*current.n_subnodes)

                n_subnodes_samples[0] = end_p - start # [start, end_p)
                n_subnodes_samples[1] = end - end_p   # [end_p, end)
                
                wn_subnodes_samples[0] = <DOUBLE_t>current.n_subnodes_samples[0]
                wn_subnodes_samples[1] = <DOUBLE_t>current.n_subnodes_samples[1] 

                # thresh = random from (start_p, end_p)
                current.threshold = Xf[start_p] + \
                                    rand_double(rand)*(Xf[end_p]-Xf[start_p]) # XXX

                if debug:
                    printf("_choose_split_point(): spoint[%d], range[%.1f, %.1f) thresh[%.1f]\n", end_p, Xf[start_p], Xf[end_p], current.threshold)
                    printf("                        leaf [%6d-%6d) n=%6.1f\n", start, end_p,  wn_subnodes_samples[0])
                    printf("                        right[%6d-%6d) n=%6.1f\n", end_p, end,    wn_subnodes_samples[1])


                # left <= start_p < thresh < end_i <= right

                self.criterion.update(samples_win, current, Xf)
                current.improvement = self.criterion.improvement(current.wn_subnodes_samples, 
                                                                current.n_subnodes, 
                                                                impurity, 
                                                                NO_DIFF_PRIVACY_BUDGET)

                if current.improvement > best_improvement:
                    best_improvement = current.improvement
                    best_i = idx

                idx += 1

        n_split_points = idx #XXX
        if n_split_points <= 0:
            printf("Warning, there's only %d split points for choosing, samples[%d-%d]\n", 
                    n_split_points, start, end)
            exit(1)
            return

        if epsilon > 0.0:
            # weight XXX
            idx = draw_from_exponential_mech( records, weights, n_split_points, sensitivity, epsilon, rand)
        else:
            idx = best_i

        current = &records[idx]

        # deep copy
        _copy_split(current, best)
        if debug:
            printf("f[%d]: selected split point %f, score %f\n", best.feature_index, best.threshold, best.improvement)
            printf(" n_samples: %d,\n", best.n_subnodes)
            printf(" n_samples: %d : %d\n",best.n_subnodes_samples[0], best.n_subnodes_samples[1])
            printf(" wn_samples: %f : %f\n",best.wn_subnodes_samples[0], best.wn_subnodes_samples[1])
            

        # free records
        for idx in range(end-start):
            current = &records[idx]
            free(current.n_subnodes_samples)
            free(current.wn_subnodes_samples)
        free(records)

    cdef SIZE_t _choose_split_feature(self,
         #       SplitRecord* best, 
                SplitRecord* records,
                SIZE_t size,
                DOUBLE_t epsilon):# nogil:

        cdef debug = 0
        if debug:
            printf("get into exp _choose_split_feature, e=%f\n", epsilon)

        cdef UINT32_t* rand = &self.rand_r_state
        cdef DOUBLE_t sensitivity = self.criterion.sensitivity

        cdef SIZE_t index = 0      
        index = draw_from_exponential_mech(
                records, 
                NULL, # weights
                size,
                sensitivity,
                epsilon,
                rand)
        
        # best[0] = records[index]
        return index

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
        # y = np.copy(y) # XXX why copy?
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
        cdef SIZE_t max_n_feature_values = 0
        cdef SIZE_t avg_n_feature_values = 0 
        cdef SIZE_t n_continuous_features = 0

        for i in range(n_features):
            if meta.discretize:
                features[i].type = FEATURE_DISCRETE
                features[i].n_values = meta.features_[i].n_values
            else:
                if meta.features_[i].type == FEATURE_DISCRETE:
                    features[i].type = FEATURE_DISCRETE
                    features[i].n_values = meta.features_[i].n_values
                else:
                    features[i].type = FEATURE_CONTINUOUS
                    features[i].n_values = 2
                    features[i].max = meta.features_[i].max_
                    features[i].min = meta.features_[i].min_
 
            if features[i].n_values > max_n_feature_values:
                max_n_feature_values = features[i].n_values
            if features[i].type == FEATURE_CONTINUOUS:
                n_continuous_features += 1
            avg_n_feature_values += features[i].n_values

        avg_n_feature_values /= n_features

        if 0:
            printf("features:\n")
            for i in range(n_features):
                if features[i].type == FEATURE_DISCRETE: 
                    printf("[%d] discrete, n %d\n", i, features[i].n_values)
                if features[i].type == FEATURE_CONTINUOUS:
                    printf("[%d] continuous, n %d, %f - %f\n",i, features[i].n_values, features[i].min, features[i].max)

        # set data
        cdef Data* data = <Data*>calloc(1,sizeof(Data))

        data.X = <DTYPE_t*> X.data
        data.y = <DOUBLE_t*> y.data
        if sample_weight is None:
            data.sample_weight = NULL
        else:
            data.sample_weight = <DOUBLE_t*> sample_weight.data

        data.X_sample_stride  = <SIZE_t> X.strides[0]/<SIZE_t> X.itemsize
        data.X_feature_stride = <SIZE_t> X.strides[1]/<SIZE_t> X.itemsize
        data.y_stride = <SIZE_t> y.strides[0] /<SIZE_t> y.itemsize

        # printf("data %p\n", data)

        if 0:
            for i in range(n_samples):
                if (y[i] != 0.0 and y[i] != 1.0) or y[i] != data.y[i] or (data.y[i]!=0.0 and data.y[i] != 1.0):
                    print y[i],

                    printf("%f,\t", data.y[i*data.y_stride])

        data.n_samples = n_samples
        data.weighted_n_samples = weighted_n_samples

        data.n_features = n_features
        data.features = features
        data.max_n_feature_values = max_n_feature_values
        data.n_continuous_features = n_continuous_features
        data.avg_n_feature_values  = avg_n_feature_values

        # classes
        data.n_outputs = n_outputs
        # data.classes   = classes
        data.n_classes = n_classes
        data.max_n_classes = max_n_classes

        self.data = data

        self.n_features = self.data.n_features
        self.n_outputs  = self.data.n_outputs
        self.classes  = []
        self.n_classes= []
        for k in range(self.n_outputs):
            classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes.append(classes_k)
            self.n_classes.append(classes_k.shape[0])

    def __dealloc__(self):
        free(self.data)
        

cdef class NBTreeBuilder:

    def __cinit__(self, 
                    SIZE_t diffprivacy_mech,
                    DOUBLE_t budget,
                    Splitter splitter,
                    SIZE_t max_depth,
                    SIZE_t max_candid_features,
                    SIZE_t min_samples_leaf,
                    object random_state,
                    bint   print_tree,
                    bint   is_prune,
                    double CF
                    ):

        self.diffprivacy_mech = diffprivacy_mech
        self.budget = budget

        self.splitter = splitter

        self.max_depth = max_depth  # verified by Classifier
        self.max_candid_features = max_candid_features # verified
        self.min_samples_leaf = min_samples_leaf

        # self.tree = None    # XXX ??? pass parameter from python
        # self.data = None
        
        self.random_state = random_state
        self.rand_r_state = random_state.randint(0, RAND_R_MAX)
        
        self.print_tree = print_tree
        self.is_prune = is_prune
        self.CF = CF

    # cpdef build(self):
    cpdef build(self,
                Tree    tree,
                DataObject dataobject,
                bint     debug):
       
        cdef Data* data = dataobject.data
        cdef UINT32_t* rand = &self.rand_r_state
        cdef Splitter splitter = self.splitter
       
        # set parameter for building tree 
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t max_candid_features = self.max_candid_features
        cdef SIZE_t min_samples_leaf =  self.min_samples_leaf

        if min_samples_leaf <= 0:
            min_samples_leaf = data.avg_n_feature_values
        if 1:
            printf("# min_samples_leaf=%d\n", min_samples_leaf)

        cdef bint print_tree = self.print_tree
        cdef bint is_prune = self.is_prune
        if debug:
            printf("Init capacity of tree\n")

        # Initial capacity of tree
        cdef SIZE_t init_capacity
        if max_depth <= 10:
            init_capacity = (2 ** (max_depth + 1)) - 1
        else:
            init_capacity = 2047
        if debug:
            printf("Tree init_capacity: %d \n", init_capacity) 
        tree._resize(init_capacity)
        self.tree = tree

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
        
        printf("# e=%f\n", epsilon_per_action)

        # ===================================================
        # recursively depth first build tree 
        if debug:
            printf("split init\n")
        splitter.init(data) # set samples_win, features_win
        cdef SplitRecord* split_record 

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
        cdef DOUBLE_t noise_n_node_samples
        cdef DOUBLE_t weighted_n_node_samples
        cdef SIZE_t node_id

        cdef DOUBLE_t impurity

        # init Stack structure
        if debug:
            printf("stack init\n")
        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record
        # push root node into stack
        rc = stack.push(0,              # start
                        data.n_samples, # end
                        0,              # depth
                        _TREE_UNDEFINED,# parent
                        0,              # index
                        data.n_features)# n_node_features

        if rc == -1:
            raise MemoryError()

        cdef bint is_leaf = 0

        cdef int pad
        if 1:
            while not stack.is_empty():

                stack.pop(&stack_record)
                start = stack_record.start
                end   = stack_record.end
                depth = stack_record.depth
                parent= stack_record.parent
                index = stack_record.index
                n_node_features = stack_record.n_node_features 

                n_node_samples = end - start
                
                # reset class distribution based on this node
                splitter.node_reset(start, end, &weighted_n_node_samples)

                impurity = splitter.node_impurity()
                 
                is_leaf = (depth >= max_depth or n_node_features <= 0)
                if epsilon_per_action > 0.0:
                    noise_n_node_samples = <DOUBLE_t>n_node_samples + noise(epsilon_per_action, rand) # XXX
                    if debug:
                        printf("noise_n_node_samples %f, max_n_f_values %u, max_n_classes %u\n",
                                noise_n_node_samples, data.max_n_feature_values, data.max_n_classes)

                    is_leaf = is_leaf or noise_n_node_samples/(data.max_n_feature_values*data.max_n_classes) < sqrt(2.0)/epsilon_per_action

                else:
                    noise_n_node_samples = <DOUBLE_t> n_node_samples 
                    is_leaf = is_leaf or ( n_node_samples <= min_samples_leaf )

                if not is_leaf:
                    # with gil:
                    if 1:
                        split_record = splitter.node_split(&n_node_features, impurity, diffprivacy_mech,  epsilon_per_action )
                        if split_record != NULL:
            
                            is_leaf = is_leaf or (split_record.feature_index == -1)
                      
                            # XXX no improvement will be made if split this node, so let it be a leaf
                            #if split_record.improvement <= 0.0:
                            #    is_leaf = True
                            #    if debug:
                            #        printf("cancel to split in f[%d]\n",split_record.feature_index)
                        else:
                            is_leaf = True
                if is_leaf:
                    if 0:
                        printf("===========================================================\n")
                        printf(" Leaf Node, depth[%u], parent[%u], index[%u]\n sample from %u to %u\n",
                                depth, parent, index, start, end)
                        printf("===========================================================\n")
                    
                    # leaTruee
                    node_id = tree._add_node(
                            parent,
                            index,
                            1,      # leaf node
                            NO_FEATURE,
                            NO_THRESHOLD,
                            0,                      # no children
                            n_node_samples,
                            weighted_n_node_samples, # xxx
                            noise_n_node_samples
                            )

                    # store class distribution into node.values
                    splitter.node_value(tree.value+node_id*tree.value_stride)
                    # add noise to the class distribution
                    if epsilon_per_action > 0.0:
                        noise_distribution(epsilon_per_action, tree.value+node_id*tree.value_stride, data, rand)
                    
                    
                    if debug and print_tree and node_id != 0:
                        for pad in range(depth):
                            printf(" | ")
                        printf("f[%2u]=%2u n=%3u\t", tree.nodes[parent].feature, index, n_node_samples)
                        splitter.criterion.print_distribution(NULL)
                        printf("\t\t")
                        splitter.criterion.print_distribution(tree.value+node_id*tree.value_stride)
                        printf("\n",n_node_samples)
                else:

                    if 0:
                        printf("===========================================================\n")
                        printf("Inner node: Depth[%u], parent[%u] index[%u]\n", depth, parent, index)
                        printf(" samples from %u to %u\n", start, end)
                        printf("===========================================================\n")
 
                    node_id = tree._add_node(
                            parent,
                            index,
                            0,     # not leaf node
                            split_record.feature_index,
                            split_record.threshold,
                            split_record.n_subnodes,  # number of children
                            n_node_samples, 
                            weighted_n_node_samples, # XXX
                            noise_n_node_samples
                            )
                    if debug:
                        printf("Selected feature[%u] to split \n\tn_children[%u], n_node_samples[%u]\n\n",
                                split_record.feature_index, split_record.n_subnodes,
                                n_node_samples,)
                    
                    if debug and print_tree:
                        if node_id != 0: 
                            for pad in range(depth):
                                printf(" | ")
                            printf("f[%u]=%u n=%u\n", tree.nodes[parent].feature, index, n_node_samples)
                    # push children into stack
                    start_i = 0 
                    end_i   = start
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
                        if 0:
                            printf("\tChild[%u] from %u to %u, left %u features\n",
                                index,start_i,end_i,n_node_features)
                        if rc == -1:
                            printf(" Error: stack.push failed\n")
                            break
                    if end_i != end:
                        printf("Error: end_i %u should be equals to end %u\n", end_i, end)
                        printf("FYI: start %u\n",start)
                        exit(1)
                    
                    free(split_record.n_subnodes_samples)
                    free(split_record.wn_subnodes_samples)
                    free(split_record)

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        
        if rc == -1:
            raise MemoryError()

        if is_prune:
            tree.calibrate_n_node_samples(0, tree.nodes[0].noise_n_node_samples)
            tree.calibrate_class_distribution( 0 )

        if print_tree:
            printf("-----------------------------------------------\n")
            printf("Before Pruning\n")
            printf("-----------------------------------------------\n")
            tree.print_tree()

        if is_prune:
            #printf("prune begins\n")
            tree.prune(0, self.CF)
            #printf("prune ends\n")

        if print_tree:
            printf("-----------------------------------------------\n")
            printf("After Pruning\n")
            printf("-----------------------------------------------\n")
            tree.print_tree()
         

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
            DataObject dataobject,
            bint debug):
                #Feature*       features,
                #SIZE_t         n_features,
                #np.ndarray[SIZE_t, ndim=1]  n_classes,
                #SIZE_t         n_outputs):

        cdef Data* data = dataobject.data
        self.data = data
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
    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) :#nogil:
       
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity
 

        # XXX no safe_realloc here because we need to grab the GIL
        # realloc self.nodes
       # cdef void* ptr = realloc(self.nodes, capacity * sizeof(Node))
       # if ptr == NULL:
       #     return -1
       # self.nodes = <Node*> ptr
        
        self.value = safe_realloc(&self.value, capacity*self.value_stride)
        self.nodes = safe_realloc(&self.nodes, capacity)
 
       # ptr = realloc(self.value,
       #               capacity * self.value_stride * sizeof(double))
       # if ptr == NULL:
       #     return -1
       # self.value = <double*> ptr
        
        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 
                    0,
                   (capacity - self.capacity) * self.value_stride * sizeof(double))
        
        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity
 
        self.capacity = capacity

        return 0

    cdef SIZE_t _add_node(self, 
                          SIZE_t parent, 
                          SIZE_t index,
                          bint   is_leaf,
                          SIZE_t feature, 
                          DOUBLE_t threshold, 
                          SIZE_t n_children,
                          SIZE_t n_node_samples, 
                          DOUBLE_t weighted_n_node_samples,
                          DOUBLE_t noise_n_node_samples
                          ):#nogil:
        """Add a node to the tree.
        The new node registers itself as the child of its parent.
        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return <SIZE_t>(-1)

        cdef Node* node = &self.nodes[node_id]
        
        if 0:
            printf("self.value is %p\n",self.value)
            printf("self.nodes is %p, node_id is %u\n",self.nodes, node_id)

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
            memset(node.children, 0, n_children*sizeof(SIZE_t))
            if node.children == NULL: # XXX error
                return <SIZE_t>(-1)
        if 0:
            printf("add node 1\n")
            test_data_y(self.data)
 

        if parent != _TREE_UNDEFINED:
            if self.nodes[parent].n_children <= index:
                # with gil:
                raise ValueError("child's index %d is greater than parent's n_classes %d"%(index, self.nodes[parent].n_children))
            self.nodes[parent].children[index]=node_id
        
        if 0:
            printf("add node 2\n")
            test_data_y(self.data)
 
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        node.noise_n_node_samples = noise_n_node_samples

        if 0:
           # printf("add node 3\n")
            printf("self.nodes is %p\n", self.nodes)
            printf("node n_node_samples %p",&(self.nodes[node_id].n_node_samples))
            test_data_y(self.data)
 

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

        cdef SIZE_t nid

        cdef SIZE_t nnid
        cdef SIZE_t j

        cdef bint debug = 0
    
        if debug:
            printf("array of node ids\n")
        if 1:
        # with nogil:
            for i in range(n_samples):
                node = self.nodes
                nid  = 0

                while not node.is_leaf:
                    if self.features[node.feature].type == FEATURE_CONTINUOUS:
                        if X[i, node.feature] < node.threshold:
                            node = &self.nodes[node.children[0]]
                        else:
                            node = &self.nodes[node.children[1]]
                    else:
                        if debug:
                            printf("node[%u] leaf:%u, goto %u subnode,\n", nid, node.is_leaf, <SIZE_t>X[i,node.feature])
                        nnid = node.children[<SIZE_t>X[i,node.feature]]
                        if nnid == 0:
                            printf("Bug node[%u], is_leaf[%u], n_children[%u]\n",nid, node.is_leaf, node.n_children)
                            printf("X[i,f]=%f\n", X[i,node.feature])
                            for j in range(node.n_children):

                                printf("%u, ",node.children[j])
                            printf("\n")
                            exit(1) 
                        nid = nnid
                        node = &self.nodes[nid]
                           


                id_data[i] = nid
                if debug:
                    printf("Find leaf node: %u,\n\n ",id_data[i])

        return node_ids

    
    cdef void calibrate_n_node_samples(self, SIZE_t node_id, DOUBLE_t fixed_n_node_samples):
        ''' top down calibrate n_node_samples of each nodes'''

        cdef Node* nodes = self.nodes
        cdef SIZE_t child_id 
        cdef Node* child

        cdef DOUBLE_t total = 0.0
        cdef SIZE_t i

        cdef Node* parent = &nodes[node_id]
        parent.noise_n_node_samples = fixed_n_node_samples
        
        if parent.is_leaf:
            return

        for i in range(parent.n_children):
            child = &nodes[parent.children[i]]
            total += child.noise_n_node_samples

        if total == 0.0:
            for i in range(parent.n_children):
                child = &nodes[parent.children[i]]
                printf("%f, ", child.noise_n_node_samples)
            printf("\n")
            exit(0)

        for i in range(parent.n_children):
            child_id = parent.children[i]
            child = &nodes[child_id]
            self.calibrate_n_node_samples( child_id, (child.noise_n_node_samples/total)*parent.noise_n_node_samples )
        
    cdef void calibrate_class_distribution(self,SIZE_t node_id):
        ''' buttom up calibarate class distribution '''

        cdef Node* nodes = self.nodes
        cdef double* value = self.value
        cdef SIZE_t value_stride = self.value_stride
        
        cdef SIZE_t n_outputs  = self.data.n_outputs
        cdef SIZE_t* n_classes = self.data.n_classes
        cdef SIZE_t stride     = self.data.max_n_classes

        cdef Node* node     = &nodes[node_id]
        cdef double* counts  = value + node_id * value_stride    
        
        cdef double total = 0.0
        cdef SIZE_t k, c, i

        # for leaf node
        if node.is_leaf == True:
            for k in range(n_outputs):
                total = 0.0
                for c in range(n_classes[k]):
                    total += counts[ k*stride + c]

                if total == 0.0:
                    continue

                for c in range(n_classes[k]):
                    counts[k*stride + c] = node.noise_n_node_samples * (counts[k*stride + c] / total) 

            return

        # for inner node
        counts = value + node_id * value_stride
        for k in range(n_outputs):
            for c in range(n_classes[k]):
                counts[k*stride + c] = 0.0

        cdef double* child_counts
        for i in range(node.n_children):
            self.calibrate_class_distribution( node.children[i] )
            child_counts = value + node.children[i] * value_stride

            for k in range(n_outputs):
                for c in range(n_classes[k]):
                    counts[k*stride + c] += child_counts[k*stride + c] 

        
    cdef double n_errors(self, double* counts, double noise_n_node_samples):

        cdef SIZE_t n_outputs = self.data.n_outputs
        cdef SIZE_t* n_classes = self.data.n_classes
        cdef SIZE_t stride = self.data.max_n_classes
        
        cdef SIZE_t k, c
        cdef double total = 0.0
        cdef double max

        for k in range(n_outputs):
            max = counts[0]
            for c in range(n_classes[k]):
                if counts[c] > max:
                    max = counts[c]
            error = noise_n_node_samples - max 
            total += error
            counts += stride

        total /= n_outputs
        return total if total > 0.0 else 0.0

    cdef double leaf_error(self, SIZE_t node_id, double CF):
      
        cdef Node* node = &self.nodes[node_id]
         
        cdef double noise_n_node_samples = node.noise_n_node_samples
        if noise_n_node_samples <= 0.0: # XXX
            return 0.0 

        cdef double* counts = self.value + node_id * self.value_stride
        cdef double error = self.n_errors( counts, noise_n_node_samples)

        return error + add_errs( noise_n_node_samples, error, CF)

    cdef double node_error(self, SIZE_t node_id, double CF):
       
        cdef Node* node = &self.nodes[node_id]
        if node.is_leaf:
            return self.leaf_error(node_id, CF)

        cdef double noise_n_node_samples = node.noise_n_node_samples
        if noise_n_node_samples <= 0.0: # XXX
            return 0.0

        cdef double errors = 0.0
        for i in range(node.n_children):
            errors += self.node_error(node.children[i], CF)
        return errors

    cdef void prune(self, SIZE_t node_id, double CF):

        cdef Node* node = &self.nodes[node_id]

        if not node.is_leaf:
            for i in range(node.n_children):
                self.prune(node.children[i], CF)
        else:
            return

        # error if it's a leaf
        cdef double leaf_error = self.leaf_error(node_id, CF)
        # error if it's a inner node
        cdef double inner_error = self.node_error(node_id, CF)
        cdef bint debug = 0
        
        if leaf_error <= inner_error + 0.1:
            if debug:
                printf("N[%d] %f <= %f +0.1\t prune\n", node_id, leaf_error, inner_error)
            # make it as leaf node
            node.is_leaf = True
            node.n_children = 0
            node.feature = -1
            node.threshold = -INFINITY
            free(node.children)
            node.children = NULL
        else:
            if debug:
                printf("N[%d] %f > %f +0.1\t unprune\n", node_id, leaf_error, inner_error)

    cdef void print_tree(self ):
       
        cdef Node* nodes = self.nodes
        cdef Node* node = &nodes[0]
        cdef SIZE_t i = 0
        for i in range(node.n_children):
            self.print_node( node.children[i], node.feature, i, 0)

    cdef void print_node(self, SIZE_t node_id, SIZE_t feature, SIZE_t index, SIZE_t depth):
    
        cdef Node* node = &self.nodes[node_id]
        cdef SIZE_t i
        
        for i in range(depth):
            printf(" | ")
        printf("n=[%6.1f] f[%2d]=[%2d] [", node.noise_n_node_samples, feature, index)
        
        cdef SIZE_t n_outputs = self.data.n_outputs
        cdef SIZE_t* n_classes = self.data.n_classes
        cdef SIZE_t stride = self.data.max_n_classes
        cdef double* counts = self.value + node_id*self.value_stride

        for k in range(n_outputs):
            for c in range(n_classes[k]):
                printf("%5.1f, ", counts[c])
            counts += stride
        printf("]\t")

        cdef double CF = 0.25 
        cdef double le = self.leaf_error(node_id, CF)
        cdef double ne = self.node_error(node_id, CF)
        cdef char op
        if le <= ne + 0.1:
            op = '<'
        else:
            op = '>'

        if not node.is_leaf:
            printf("[%3.2f %c %3.2f]", le, op, ne)
        else:
            printf("[%3.2f]", le)

        printf("\n")


        if not node.is_leaf:
            for i in range(node.n_children):
                self.print_node( node.children[i], node.feature, i, depth+1)

      


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
    (Node*)

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
cdef DOUBLE_t laplace(DOUBLE_t b, UINT32_t* rand) except -1: # nogil
    # No diffprivacy
    if b <= 0.0:
        return 0.0

    cdef DOUBLE_t uniform = rand_double(rand)-0.5 # gil
    if uniform > 0.0:
        return -b*log(1.0-2*uniform)
    else:
        return +b*log(1.0+2*uniform) 

cdef DOUBLE_t noise(DOUBLE_t epsilon, UINT32_t* rand): # nogil:

    cdef bint debug = 0

    if epsilon == NO_DIFF_PRIVACY_BUDGET:
        return 0.0
    
    # printf("Error, should not be shown")
    cdef DOUBLE_t noise = 0.0

    noise =  laplace(1.0/epsilon, rand)
    if debug:
        printf("e=%f, noise=%f\n", epsilon, noise)
    return noise

cdef void noise_distribution(DOUBLE_t epsilon, DOUBLE_t* dest, Data* data, UINT32_t* rand): #nogil:

    cdef SIZE_t k = 0
    cdef SIZE_t i = 0
    
    cdef SIZE_t n_outputs = data.n_outputs
    cdef SIZE_t* n_classes = data.n_classes
    cdef SIZE_t stride = data.max_n_classes
    cdef SIZE_t zero_cnt = 0

    if epsilon <= 0.0:
        return

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


cdef void normalize(double* dist, SIZE_t size):

    cdef double total = 0.0
    
    
    cdef SIZE_t i = 0
    for i in range(size):
        if dist[i] < 0.0:
            raise("dist[%d]=%f should be greater than 0"%(i,dist[i]))
            exit(1)
        total += dist[i] 

    if total == 0.0:
        return

    for i in range(size):
        dist[i] = dist[i]/total


cdef SIZE_t draw_from_distribution(DOUBLE_t* dist, SIZE_t size, UINT32_t* rand) except -1: #nogil
    """ numbers in arr should be greater than 0 """

    cdef bint debug = 0
    cdef DOUBLE_t total = 0.0
    cdef DOUBLE_t point = 0.0 
    cdef DOUBLE_t current = 0.0 
    cdef SIZE_t i = 0

    normalize(dist, size)

    for i in range(size):
        total += dist[i] 

    # if numbers in arr all equal to 0
    if total == 0.0:
        return rand_int(size, rand)

    point = rand_double(rand)
    if debug:
        printf("total is %f,  random %f from (0,1)\n", total,  point)

    if debug: 
        for i in range(size):
            printf("%f, ", dist[i])
        printf("\n")

    i = 0
    current = 0.0
    for i in range(size):
        current += dist[i]
        if current > point:
            return i

    return size-1

cdef SIZE_t correct_count[3]
correct_count[0] = 0
correct_count[1] = 0
correct_count[2] = 0

cdef SIZE_t draw_from_exponential_mech( 
            SplitRecord* records, 
            double* weights, 
            int size, 
            double sensitivity, 
            double epsilon, 
            UINT32_t* rand) except -1: #nogil 

    cdef double* improvements 
    #cdef double* improvements_minusmax 
    cdef double max_improvement = -INFINITY
    cdef int best_i
    cdef int ret_idx = 0

    cdef int i = 0
    cdef int j = 0
    cdef double w = 0.0

    #epsilon = 4.0 #XXX
    cdef bint debug = 0
    if debug:
        printf("draw_from_exponential_mech: e=%f s=%f n=%d\n", epsilon, sensitivity, size)

    improvements = <double*>calloc(size, sizeof(double))
    #improvements_minusmax = <double*>calloc(size,sizeof(double))
    #normalized_improvements = <double*>calloc(size,sizeof(double))
    
    i = 0
    for i in range(size):
        improvements[i] = records[i].improvement
    #    normalized_improvements[i] = records[i].improvement
        if improvements[i] > max_improvement:
            max_improvement = improvements[i]
            best_i = i

    #normalize(normalized_improvements, size)
    # rescale from 0 to 1
    i = 0
    for i in range(size):
        if debug:
            printf("%2d: f[%2d] %.2f\t",i, records[i].feature_index, improvements[i])

        if weights == NULL:
            w = 1.0
        else:
            w = weights[i]

        #improvements_minusmax[i] = improvements[i] -  max_improvement
        improvements[i] = improvements[i] -  max_improvement
        improvements[i] = w * exp(improvements[i]*epsilon/(2*sensitivity))
        if debug:
            printf("%.2f\t", improvements[i])
            #printf("%f\t", improvements_minusmax[i])
    
            printf("\n")

        #improvements_minusmax[i] = exp(improvements_minusmax[i]*epsilon/(2*sensitivity))
        #if debug:
        #    printf("%f\t" ,improvements_minusmax[i])

        #if debug:
        #    printf("%f\n", normalized_improvements[i])
        # normalized_improvements[i] = exp(normalized_improvements[i]*epsilon/(2*sensitivity))
        #if debug:
        #    printf("%f\n", normalized_improvements[i])
    ret_idx = draw_from_distribution(improvements, size, rand)
    #i = draw_from_distribution(improvements_minusmax, size, rand)
    #j = draw_from_distribution(normalized_improvements,size,rand)
    

    correct_count[0] += 1
    if ret_idx == best_i:
        correct_count[1] += 1
    #if i == best_i:
    #    correct_count[1] += 1
    #if j == best_i:
    #    correct_count[2] += 1
    free(improvements)

    if debug:
        printf("selected best %d,   %d \n", best_i, ret_idx ) 
        printf("correct %d : %d : %d\n", correct_count[0], correct_count[1], correct_count[2] )
        printf("\n")
    return ret_idx

# for pruning
cdef double add_errs(double N, double e, double CF):

    cdef double base
    if e < 1.0:
        base = N*(1- pow(CF, 1.0/N))
        
        if e == 0.0:
            return base
        return base + e * ( add_errs( N, 1.0, CF) - base )

    if e + 0.5 >= N:
        return N-e if N-e > 0.0 else 0.0

    cdef double z = norm.ppf( 1-CF )
    cdef double f = (e + 0.5) / N
    cdef double r = ( f + (z * z) / (2 * N) + z * sqrt((f / N) - (f * f / N) + (z * z / (4 * N * N)))
                    )/ (1 + (z * z) / N)

    return (r * N) - e;
 
# ===============================================================
# For test
# ==============================================================

cdef void test_data_y( Data* data): # nogil:
    
    cdef SIZE_t i
    cdef SIZE_t j
   
    cdef DOUBLE_t value

    for i in range(data.n_samples):
        value = data.y[i]
        if value > 1.0 or value < 0.0:
            printf("Error: data_y test failed, y_stride is %u, size of double[%u]\n", data.y_stride, sizeof(DOUBLE_t))
            printf("address of y is %p\n", data.y)
            for j in range(data.n_samples):        
                value = data.y[j]
                if value > 1.0 or value < 0.0: 
                    printf("[%u] %f, %d, %x,\t", j, value, value, value)
                    printf("[%u] %f, %d, %x,\t", j, value, value, value)
                    printf("[%u] %f, %d, %x,\t", j, value, value, value)
                    printf("[%u] %f, %d, %x,\t", j, value, value, value)
            printf("\n")            
            exit(1) 
        


