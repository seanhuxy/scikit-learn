import numpy as np
from scipy.io import arff
from scipy.cluster.vq import kmeans2, vq
import numbers
import copy

FEATURE_CONTINUOUS = 0
FEATURE_DISCRET = 1
       
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
                    raise Exception, "feature({0}) has no value {1}".format( str(self), vector[i])
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
        self.is_discretized = False

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

        self.is_discretized = True
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

        data, meta = arff.loadarff(arff_file)
        names = meta.names()
        types = meta.types()

        features = []
        for name in names:
            feature = Feature()
            feature.name = name
            
            type, values = meta[name] 
            if type is "numeric":
                feature.type = FEATURE_CONTINUOUS
                
            elif type is "nominal":
                feature.type = FEATURE_DISCRET
                feature.values = []
                feature.indices = {}
                for v in values:
                    feature.indices[v] = len(feature.values)
                    feature.values.append(v)

                feature.n_values = len(feature.values)

            else:
                raise Exception, "Error: parsing {0}, {1} ".format(name, meta[name])

            features.append( feature )
        
        new_data = np.zeros(( len(data), len(features) ))
        
        row_i = 0
        for row in data:
            col_i = 0 
            for name in names:

                v = row[name]
                
                feature = features[col_i]
                if feature.type == FEATURE_CONTINUOUS:
                    new_data[row_i, col_i] = v 
                    if feature.max in [None, np.NaN]:
                        feature.max = v
                    if feature.min in [None, np.NaN]:
                        feature.min = v
                    feature.max = v if v > feature.max else feature.max
                    feature.min = v if v < feature.min else feature.min

                elif feature.type == FEATURE_DISCRET:
                    if not feature.indices.has_key(v):
                        raise Exception, "Error feature {0} has not value {1}".format(str(feature), v)

                    new_data[row_i, col_i] = feature.indices[v]

                else:
                    raise Exception, "Error feature type: {0}".format(str(feature))

                col_i +=1

            row_i += 1
            
        self.features = features
        self.data = new_data
        self.is_transferred = True #XXX

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

    def load(self, feature_file, data_file=None):
        if feature_file.endswith("arff"):
            self.load_arff(feature_file)
        else:
            self.load_raw(feature_file, data_file)
            self.transfer()
        return self

        


