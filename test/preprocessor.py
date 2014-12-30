import numpy as np
from scipy.cluster.vq import kmeans2, vq
import numbers
import copy

FEATURE_CONTINUOUS = 0
FEATURE_DISCRET = 1

#class Feature():
#
#    def __init__(self):
#        self.name = None    # string 'city'
#        self.type = None
#
#        self.n_values = 0
#        
#        # for discret feature
#        self.indices = {}   # dict  { 'beijing' : 0, 'shanghai': 1 } 
#        self.values  = []   # list  [ 'beijing', 'shanghai' ]
#       
#        # for continuous feature
#        self.max = None 
#        self.min = None
#        self.interval = np.NaN
#
#        self.centeriods = None
#
#    # kmeans cluster
#    def discretize(self, value):
#       
#        if isinstance(value, (numbers.Integer, numbers.real, np.float, np.int)):
#            value = np.array([value])
#    
#        if self.centeriods is None:
#            print "Warning, feature {0} centeriods is None".format(self.name)
#
#        ret, _ = vq( value, self.centeriods)
#
#        return ret
#
#    #def discretize(self, v):
#    #     
#    #    try:
#    #        index = int( (v - self.min)/ self.interval )
#    #    except ValueError:
#    #        print v, self.min, self.max 
#    #    if index >= self.n_values:
#    #        if index == self.n_values and v == self.max:
#    #            index = self.n_values-1
#    #        else:
#    #            error = "Discretizing continous value %f error, min %f, max %f, bins %u"%(v, self.min, self.max, self.n_values)
#    #            raise ValueError(error)
#
#    #    return index
#
#    def __str__(self):
#        if self.type == FEATURE_DISCRET:
#            ret = 'name {0}\tn_values {1}\nindices {2}\nvalues {3}'.format(self.name, self.n_values, self.indices, self.values)
#        else:
#            ret = 'name {0}\tn_values {1}\tmax {2}\tmin {3}'.format(self.name, self.n_values, self.max, self.min)
#        
#        return ret 
#

#class Preprocessor:
#
#    def __init__(self):
#        pass
#
#    def _load_data_file(self, features, data_file ):
#        try:
#            data = np.loadtxt( data_file )
#            if data.ndim != 2:
#                raise Exception, "data is not 2-dimension array"
#
#            n_samples, n_features = data.shape[0], data.shape[1]
#            if n_features != len(features):
#                raise Exception, "number of features {0} in data \
#                            is not consistent with that {1} in feature file".format(n_features, len(features))
#        except:
#            print "Error occurs in load_data_file()"
#
#        return data
#
#    def _read_meta(self, features, data):
#
#        for record in data:
#            for f in range(len(features)):
#                feature = features[f]
#                value   = record[f]
#
#                if feature.type == FEATURE_DISCRET:
#                    if feature.values is None:
#                        feature.values = []
#                    if feature.indices is None:
#                        feature.indices = {}
#
#                    if value not in feature.values:
#                        feature.indices[value] = len(feature.values)
#                        feature.values.append(value)
#                        feature.n_values = len(feature.values)
#
#                elif feature.type == FEATURE_CONTINUOUS:
#                    if feature.max in [None, np.NaN]:
#                        feature.max = value
#                    else:
#                        feature.max = value if value > feature.max else feature.max
#
#                    if feature.min in [None, np.NaN]:
#                        feature.min = value
#                    else:
#                        feature.min = value if value < feature.min else feature.min
#
#                else:
#                    raise Exception, "Unknown feature type", feature.type
#
#
#    def _load_one_feature(self, line ):
#
#        feature = Feature()
#        try:
#            strings = line.split(" ",2)
#            feature.name = strings[0]
#            type = strings[1]
#            if type == "discret":
#                feature.type = FEATURE_DISCRET
#                
#            elif type == "continuous":
#                feature.type = FEATURE_CONTINUOUS
#            else:
#                raise Exception, "Unexcepted feature type {0} ".format(type)
#
#            return feature
#
#        except Exception, e:
#            print "Error occurs when parsing feature: ", line
#            print e
#
#    def _load_feature_file(self, feature_file ):
#        lines = []
#        with open(feature_file) as f:
#            lines = f.readlines()
#        
#        features = []
#        for line in lines:
#            feature = self._load_one_feature( line )
#            features.append(feature)
#
#        return features
#
#
#    def _export_feature(self, features, path):
#        
#        try:
#            #with open(path) as of:
#            of = open(path, 'w')
#            for f in features:
#                of.write("{0} ".format(f.name))
#                if f.type == FEATURE_DISCRET:
#                    of.write("discret ")
#                    of.write("{0}".format(f.n_values)) 
#                else:
#                    of.write("continuous ")
#                    of.write("{0} {1}".format(f.min, f.max))
#
#                of.write("\n")
#            of.close()
#        except:
#            print "Error occurs when exporting features to file {0}".format(path)
#
#    def _export_data(self, data, path):
#        try:
#            of = open(path, 'w')
#            for row in data:
#                for v in row:
#                    of.write("{0} ".format(v))
#                of.write("\n")
#            of.close()
#        except:
#            print "Error occurs when exporting data to file {0}".format(path)
#    
#    def discretize(self, nbins, method="cluster"):
#        features = self.features_
#        data = self.data_
#
#        #newobject = copy.deepcopy(self)
#
#        for i in range(len(features)):
#            f = features[i]
#            if f.type != FEATURE_CONTINUOUS:
#                continue
#            values = np.take(data, i, axis = 1)
#            
#            if method == "cluster":
#                f.centeriods = kmeans2( values, nbins, iter=20) 
#                f.centeriods = sort(f.centeriods)
#            
#            else:
#                if f.max in [None, np.nan]:
#                    f.max = np.max( values )
#                if f.min in [None, np.nan]:
#                    f.min = np.min( values )
#                   
#                interval = (f.max - f.min) /nbins
#
#                f.centeriods = np.zeros(nbins)
#                for j in range(bins):
#                    f.centeriods[j] = (j+0.5)*interval
#        
#        return self
#
#    def export(self, data_file,  feature_file):
#        ''' export to new file '''
#        self._export_feature( self.features, feature_file)
#        self._export_data( self.data, data_file)
#
#    def load(self, data_file, feature_file, is_read_meta_from_data=True, is_discretize=False, nbins=10):
#
#        ''' import from exist file ''' 
#        features= self._load_feature_file( feature_file )
#        data    = self._load_data_file( features, data_file )
#        
#        if is_read_meta_from_data:
#            self._read_meta(features, data)        
#       
#        if is_discretize:
#            discretize(features, data, nbins)
#
#        self.data_ = data
#        self.features_ = features
#        return data, features
        
class Feature:
    
    def __init__(self):
        self.name = None
        self.type = None

        self.n_values = 0

        self.values = []
        self.indices = {}

        self.max = None
        self.min = None

        self.centeriods = None

        self.is_discreted = False

    def transfer(self, vector):
        ''' transfor raw value 
            discret
            { Beijing, Shanghai, Guangzhou } -> { 0, 1, 2}
            { 48, 52 } -> { 0, 1}

            continuous:
            { 20, 30, 40 } -> { 2, 3, 4}
        '''
        if isinstance(vector, (numbers.Integral, numbers.Real, np.float, np.int)):
            vector = np.array([vector])


        if self.type == FEATURE_DISCRET:
            ret = np.copy(vector)

            for i in range(len(vector)):
                if self.indices.has_key(vector[i]):
                    
                    ret[i] = self.indices[vector[i]]
                else:
                    raise Exception, "feature({0}) has no value {1}".format( str(self), value)
            return ret

        else: # continuous

            if self.is_discreted:

                if self.centeriods is None:
                    raise Exception, "feature({0}) has no centeriods".format( str(self) )

                ret, _ = vq( vector, self.centeriods)
                return ret

            else:
                return vector
            

    def discretize(self, data, nbins, method = "cluster"):

        if self.type == FEATURE_DISCRET:
            return 

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
        
        return ret

    def __str__(self):
        ''' 
        continuous:
        ----------
        age continuous 0 90

        discretized continous:
        ----------------------
        age continuous discretized 10 5 15 25 35 45 55 65 75 85 95

        discret: (represent by string)
        ------------------------------
        city discret 4 beijing shanghai shenzhen guangzhou

        discret: (represent by number)
        zip discret 3 010 020 030
        '''

        if self.type == FEATURE_CONTINUOUS:
            
            ret = "{0} {1} ".format(self.name, "continuous")
            if not self.is_discreted:
                ret += "{0} {1} ".format(self.min, self.max)
                return ret
            else:
                ret += "discretized {0} ".format(self.n_values)
                for c in self.centeriods:
                    ret += "{0} ".format(c)
                return ret

        if self.type == FEATURE_DISCRET:
            ret = "{0} {1} {2} ".format(self.name, "discret", self.n_values)

            for v in self.values:
                ret += "{0} ".format(v)
            
            return ret

        return "error feature"

    def parse(self, string):
        ''' parse a string, to get the feature's meta data '''
        string = string.strip('\n')

        strings = string.split()

        if len(strings) < 2:
            raise Exception, "Feature(): parsing {0}, expected 2 or more discriptors".format(string)

        self.name = strings[0]
        type = strings[1]
        if type == "discret":
            self.type = FEATURE_DISCRET
        elif type == "continuous":
            self.type = FEATURE_CONTINUOUS
        else:
            raise Exception, "Feature(): parsing {0}, unexpected feture type".format(string)

        curr = 2
        if len(strings) > 2:
            
            if self.type == FEATURE_CONTINUOUS:

                if strings[curr] == "discretized":
                    self.discretize = True
                    curr += 1

                    n_values = int( strings[curr] )
                    curr += 1
                    if curr + n_values < len(strings):
                        raise Exception, "parsing {0}, less centeriods then expected".format(string)

                    centeriods = []
                    for i in range(n_values):
                        v = float( strings[curr + i] )
                        centeriods.append( v )

                else:
                    self.min = float( strings[curr] )
                    self.max = float( strings[curr+1] ) 

            else:   # DISCRET

                n_values = int( strings[curr] )
                curr += 1
                if curr + n_values < len(strings):
                    raise Exception, "parsing {0}, less values then expected".format(string)

                values = []
                indices = {}
                for i in range(n_values):
                    v = strings[curr + i]

                    if v not in values:
                        indices[v] = len(values)
                        values.append( v )
                    else:
                        raise Exception, "parsing {0}, value {1} is duplicated".format(string, v)

                self.values = values
                self.indices = indices
                self.n_values = n_values


    def parse_meta_from_data(self, data):
        
        if self.type == FEATURE_DISCRET:
            if self.values is None:
                self.values = []
            if self.indices is None:
                self.indices = {}

        for v in data:

            if self.type == FEATURE_DISCRET:

                if v not in self.values:
                    self.indices[v] = len(self.values)
                    self.values.append(v)

                    self.n_values = len(self.values)

            elif self.type == FEATURE_CONTINUOUS:

                if self.max in [None, np.NaN]:
                    self.max = v
                else:
                    self.max = v if v > self.max else self.max

                if self.min in [None, np.NaN]:
                    self.min = v
                else:
                    self.min = v if v < self.min else self.min

            else:
                raise Exception, "Unknown feature type", self.type


class Preprocessor:

    def __init__(self):
	self.features = None
        self.data = None

        self.is_transferred = False

    def get_X(self):
        if self.data is None:
            print "Warning, please first input raw dataset by load() function"
            return None
        return self.data[:, :-1]

    def get_y(self):
        if self.data is None:
            print "Warning, please first input raw dataset by load() function"
            return None
        return self.data[:, -1]

    def get_features(self):
        if self.features is None:
            print "Warning, please first input raw dataset by load() function"
            return None
        return self.features

    def discretize(self, nbins, method = "cluster"):
        features = self.features
        data = self.data

        for i in range(len(features)):
            f = features[i]
            if f.type != FEATURE_CONTINUOUS:
                continue

            values = np.take(data, i, axis = 1)
            f.discretize( values, nbins, method)

        return self

    def export(self, feature_file, data_file):

        features = self.features
        data = self.data
        if features is None or data is None:
            print "Warning, please first input raw dataset by load() function"
            return None

        of = open(feature_file, 'w')
        for f in features:
            of.write(str(f))
            of.write("\n")
        of.close()

        if self.is_transferred == False:
            self.transfer()

        of = open(data_file, 'w')
        for row in data:
            for v in row:
                of.write("{0} ".format(v))
            of.write("\n")
        of.close()

        return self

    def load_arff(self, arff_file):
        
        return self

    def _read_meta(self, features, data):
        n_features = data.shape[1]
        for i in range(n_features):
            features[i].parse_meta_from_data( np.take(data, [i], axis=1).flatten() )

    def _load_features(self, feature_file):
        lines = []
        with open(feature_file) as f:
            lines = f.readlines()
        
            features = []
            for line in lines:
                feature = Feature()
                feature.parse( line )

                features.append(feature)

        return features

    def _load_data(self, features, data_file ):
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

    def transfer(self):
        if self.data is None:
            print "Waring, please first input raw dataset by load() function"
            return
        
        data = self.data
        features = self.features

        for i in range(len(features)):
            col = features[i].transfer( np.take( data, i, axis=1) )
            data[:,i] = col

        self.is_transferred = True

    def load_raw(self, feature_file, data_file):
        features = self._load_features( feature_file )
        data = self._load_data(features, data_file )

        self._read_meta( features, data)
        self.data = data
        self.features = features

        return self

    def load_existed(self, feature_file, data_file):
        features = self._load_features( feature_file )
        data = self._load_data( data_file )

        self.features = features
        self.data = data

        return self

