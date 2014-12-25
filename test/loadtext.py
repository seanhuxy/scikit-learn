import numpy as np
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

    def discretize(self, v):
         
        try:
            index = int( (v - self.min)/ self.interval )
        except ValueError:
            print v, self.min, self.max 
        if index >= self.n_values:
            if index == self.n_values and v == self.max:
                index = self.n_values-1
            else:
                error = "Discretizing continous value %f error, min %f, max %f, bins %u"%(v, self.min, self.max, self.n_values)
                raise ValueError(error)

        return index

    def __str__(self):
        if self.type == FEATURE_DISCRET:
            ret = 'name {0}\tn_values {1}\nindices {2}\nvalues {3}'.format(self.name, self.n_values, self.indices, self.values)
        else:
            ret = 'name {0}\tn_values {1}\tmax {2}\tmin {3}\tinterval {4}'.format(self.name, self.n_values, self.max, self.min, self.interval)
        
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
            for f in range(n_features):
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
                        feature.n_values = len(values)

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
                raise Exception, "Unexcepted feature type ", type

            return feature

        except:
            print "Error occurs when parsing feature: ", line

    def _load_feature_file( feature_file ):
        lines = [] \
        with open(feature_file) as f:
            lines = f.readlines()
        
        features = []
        for line in lines:
            feature = load_one_feature( line )
            features.append(feature)

        return features


    def _export_feature(features, path):
        try:
            of = open(path, 'w')
            for f in features:
                of.write("{0} {1}".format(f.name, f.type)
                if f.type == FEATURE_DISCRET:
                    of.write("{0}\n".format(f.n_values)) 
                else:
                    of.write("{0} {1}\n".format(f.min, f.max))

            of.close()
        except:
            print "Error occurs when exporting features to file {0}".format(path)

    def _export_data(data, path):
        try:
            of = open(path, 'w')
            for row in data:
                for v in row:
                    of.write("{0} ".format(v))
                of.write("\n")
            of.close()
        except:
            print "Error occurs when exporting data to file {0}".format(path)
    
    def discretize(self, nbins):
        features = self.features
        data = self.data

        newobject = copy.deepcopy(self)

        return newobject

    def export(self, feature_file, data_file):
        ''' export to new file '''
        self._export_feature( self.features)
        self._export_data( self.data)

    def load( data_file, feature_file, is_read_meta_from_data=True, is_discretize=False, nbins=10):

        ''' import from exist file ''' 
        features= self._load_feature_file( feature_file )
        data    = self._load_data_file( features, data_file )
        
        if is_read_meta_from_data:
            self._read_meta(features, data)        
       
        if is_discretize:
            discretize(features, data, nbins)
        
        return X, y, features
        

