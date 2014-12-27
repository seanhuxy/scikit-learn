import numpy as np
from scipy.cluster.vq import kmeans2, vq
import numbers
import copy

FEATURE_CONTINUOUS = 0
FEATURE_DISCRET = 1

class Feature():

    def __init__(self):
        self.name = None    # string 'city'
        self.type = None

        self.n_values = 0
        
        # for discret feature
        self.indices = {}   # dict  { 'beijing' : 0, 'shanghai': 1 } 
        self.values  = []   # list  [ 'beijing', 'shanghai' ]
       
        # for continuous feature
        self.max = None 
        self.min = None
        self.interval = np.NaN

        self.centeriods = None

    # kmeans cluster
    def discretize(self, value):
       
        if isinstance(value, (numbers.Integer, numbers.real, np.float, np.int)):
            value = np.array([value])
    
        if self.centeriods is None:
            print "Warning, feature {0} centeriods is None".format(self.name)

        ret, _ = vq( value, self.centeriods)

        return ret

    #def discretize(self, v):
    #     
    #    try:
    #        index = int( (v - self.min)/ self.interval )
    #    except ValueError:
    #        print v, self.min, self.max 
    #    if index >= self.n_values:
    #        if index == self.n_values and v == self.max:
    #            index = self.n_values-1
    #        else:
    #            error = "Discretizing continous value %f error, min %f, max %f, bins %u"%(v, self.min, self.max, self.n_values)
    #            raise ValueError(error)

    #    return index

    def __str__(self):
        if self.type == FEATURE_DISCRET:
            ret = 'name {0}\tn_values {1}\nindices {2}\nvalues {3}'.format(self.name, self.n_values, self.indices, self.values)
        else:
            ret = 'name {0}\tn_values {1}\tmax {2}\tmin {3}'.format(self.name, self.n_values, self.max, self.min)
        
        return ret 


class Preprocessor:

    def __init__(self):
        pass

    def _load_data_file(self, features, data_file ):
        try:
            data = np.loadtxt( data_file )
            if data.ndim != 2:
                raise Exception, "data is not 2-dimension array"

            n_samples, n_features = data.shape[0], data.shape[1]
            if n_features != len(features):
                raise Exception, "number of features {0} in data \
                            is not consistent with that {1} in feature file".format(n_features, len(features))
        except:
            print "Error occurs in load_data_file()"

        return data

    def _read_meta(self, features, data):

        for record in data:
            for f in range(len(features)):
                feature = features[f]
                value   = record[f]

                if feature.type == FEATURE_DISCRET:
                    if feature.values is None:
                        feature.values = []
                    if feature.indices is None:
                        feature.indices = {}

                    if value not in feature.values:
                        feature.indices[value] = len(feature.values)
                        feature.values.append(value)
                        feature.n_values = len(feature.values)

                elif feature.type == FEATURE_CONTINUOUS:
                    if feature.max in [None, np.NaN]:
                        feature.max = value
                    else:
                        feature.max = value if value > feature.max else feature.max

                    if feature.min in [None, np.NaN]:
                        feature.min = value
                    else:
                        feature.min = value if value < feature.min else feature.min

                else:
                    raise Exception, "Unknown feature type", feature.type


    def _load_one_feature(self, line ):

        feature = Feature()
        try:
            strings = line.split(" ",2)
            feature.name = strings[0]
            type = strings[1]
            if type == "discret":
                feature.type = FEATURE_DISCRET
                
            elif type == "continuous":
                feature.type = FEATURE_CONTINUOUS
            else:
                raise Exception, "Unexcepted feature type {0} ".format(type)

            return feature

        except Exception, e:
            print "Error occurs when parsing feature: ", line
            print e

    def _load_feature_file(self, feature_file ):
        lines = []
        with open(feature_file) as f:
            lines = f.readlines()
        
        features = []
        for line in lines:
            feature = self._load_one_feature( line )
            features.append(feature)

        return features


    def _export_feature(self, features, path):
        
        try:
            #with open(path) as of:
            of = open(path, 'w')
            for f in features:
                of.write("{0} ".format(f.name))
                if f.type == FEATURE_DISCRET:
                    of.write("discret ")
                    of.write("{0}".format(f.n_values)) 
                else:
                    of.write("continuous ")
                    of.write("{0} {1}".format(f.min, f.max))

                of.write("\n")
            of.close()
        except:
            print "Error occurs when exporting features to file {0}".format(path)

    def _export_data(self, data, path):
        try:
            of = open(path, 'w')
            for row in data:
                for v in row:
                    of.write("{0} ".format(v))
                of.write("\n")
            of.close()
        except:
            print "Error occurs when exporting data to file {0}".format(path)
    
    def discretize(self, nbins, method="cluster"):
        features = self.features
        data = self.data

        #newobject = copy.deepcopy(self)

        for i in range(len(features)):
            f = features[i]
            if f.type != FEATURE_CONTINUOUS:
                continue
            values = np.take(data, i, axis = 1)
            
            if method == "cluster":
                f.centeriods = kmeans2( values, nbins, iter=20) 
                f.centeriods = sort(f.centeriods)
            
            else:
                if f.max in [None, np.nan]:
                    f.max = np.max( values )
                if f.min in [None, np.nan]:
                    f.min = np.min( values )
                   
                interval = (f.max - f.min) /nbins

                f.centeriods = np.zeros(nbins)
                for j in range(bins):
                    f.centeriods[j] = (j+0.5)*interval
        
        return self

    def export(self, data_file,  feature_file):
        ''' export to new file '''
        self._export_feature( self.features, feature_file)
        self._export_data( self.data, data_file)

    def load(self, data_file, feature_file, is_read_meta_from_data=True, is_discretize=False, nbins=10):

        ''' import from exist file ''' 
        features= self._load_feature_file( feature_file )
        data    = self._load_data_file( features, data_file )
        
        if is_read_meta_from_data:
            self._read_meta(features, data)        
       
        if is_discretize:
            discretize(features, data, nbins)

        self.data = data
        self.features = features
        return data, features
        

