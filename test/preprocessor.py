import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.cluster.vq import kmeans2, vq
import numbers
import copy

from sklearn.preprocessing import Imputer
from time import time

FEATURE_CONTINUOUS = 0
FEATURE_DISCRET = 1
       
class Feature:
    
    def __init__(self):
        self.name = None
        self.type = None

        self.n_values = 0

        self.values = []    # [ 'Beijing', 'Shanghai', 'Guangzhou' ]
        self.indices = {}   # { 'Beijing': 0, 'Shanghai': 1, 'Guangzhou': 2}

        self.max = None
        self.min = None

        self.is_discretized = False

    def transfer(self, data):
        ''' transfor raw value 
            discret
            { Beijing, Shanghai, Guangzhou } -> { 0, 1, 2}
            { 48, 52 } -> { 0, 1}

            continuous:
            { 20, 30, 40 } -> { 2, 3, 4}
        '''
        if isinstance(data, (numbers.Integral, numbers.Real, np.float, np.int)):
            data = np.array([data])

        if self.type == FEATURE_DISCRET:
            
            if self.values is None:
                raise Exception, "feature({0}) has no values".format( str(self) )

            if self.is_discretized:
                data, _ = vq( data, self.values)

            else:
                for i in range(len(data)):
                    if not self.indices.has_key(data[i]):
                        raise Exception, "feature({0}) has no value {1}".format( str(self), data[i])

                    data[i] = self.indices[data[i]]

            return data

        else: # continuous
            return data
            

    def discretize(self, data, nbins, method = "cluster"):

        if self.type == FEATURE_DISCRET:
            return 

        f = self
        if method == "cluster":
            centeriods, _ = kmeans2( data, nbins, iter=20) 
            centeriods = np.unique(centeriods)
        
        else:
            max = self.max
            min = self.min

            if max in [None, np.nan]:
                max = np.max( data )
            if min in [None, np.nan]:
                min = np.min( data )
                   
            interval = (max - min) /nbins

            centeriods = np.zeros(nbins)
            for j in range(bins):
                centeriods[j] = (j+0.5)*interval

        # transfer data 
        #data, _ = vq( data, centeriods)

        f.values = centeriods
        f.n_values = len(centeriods)

        f.is_discretized = True
        f.type = FEATURE_DISCRET

        #return data
        
    def parse(self, string):
        ''' parse a string, to get the feature's meta data '''
        
        string = string.strip('\n')
        strings = string.split()

        if len(strings) < 2:
            raise Exception, "Feature(): parsing {0}, expected 2 or more discriptors".format(string)

        self.name = strings[0]
        type_ = strings[1]
        if type_ == "discret":
            self.type = FEATURE_DISCRET
        elif type_ == "continuous":
            self.type = FEATURE_CONTINUOUS
        else:
            raise Exception, "Feature(): parsing {0}, unexpected feture type".format(string)

        curr = 2
        if len(strings) > 2:
            
            if self.type == FEATURE_CONTINUOUS:

                self.min = float( strings[curr] )
                self.max = float( strings[curr+1] ) 

            else:   # DISCRET

                if strings[curr] == "discretized":
                    self.is_discretized = True
                    curr += 1

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
            values = []
            indices = {}

            values = np.unique( data )

            for i, v in enumerate(values):
                indices[ v ] = i

            self.n_values = len(values)
            self.values = values
            self.indices= indices

        elif self.type == FEATURE_CONTINUOUS:

            if len(data) <= 0:
                raise Exception, "No content in data {0}".format(data)

            #max_ = data[0]
            #min_ = data[0]

            #for v in data:
            #    max_ = v if v > max_ else max_
            #    min_ = v if v < min_ else min_

            max_ = np.max(data)
            min_ = np.min(data)

            self.max = max_ if self.max in [None, np.NaN] or max_ > self.max else self.max
            self.min = min_ if self.min in [None, np.NaN] or min_ < self.min else self.min

            #print str(self)

        else:
            raise Exception, "Unknown feature type", self.type

    def __str__(self):
        ''' 
        (1) continuous:
            e.g. age continuous 0 90

        (2) discretized continous:
            e.g. age discret discretized 10 5 15 25 35 45 55 65 75 85 95

        (3) discret: (represent by string)
            e.g. city discret 4 beijing shanghai shenzhen guangzhou 
                 zip discret 3 010 020 030 
        '''
        if self.type == FEATURE_CONTINUOUS:
            ret = "{0} {1} {2} {3}".format(self.name, "continuous", self.min, self.max)

        elif self.type == FEATURE_DISCRET:
            ret = "{0} {1} ".format(self.name, "discret" )
            if self.is_discretized:
                ret += "discretized "

            ret += "{0} ".format(self.n_values)

            for v in self.values:
                ret += "{0} ".format(v)
            
        else:
            return "error feature"

        return ret


class Preprocessor:

    def __init__(self):
        self.n_train_samples = 0
        self.n_test_samples  = 0

        self.features = None
        self.data = None

        self.is_discretized = False

    def check_load(self):
        if self.data is None or self.features is None:
            raise Exception, "Warning, please first input raw dataset by load() function"

    def get_X(self, data):
        self.check_load()
        return data[:, :-1]

    def get_y(self, data):
        self.check_load()
        return data[:, -1]

    def get_train(self):
        self.check_load()

        n_train_samples = self.n_train_samples

        data = self.data[:n_train_samples,:]
        return data

    def get_test(self):
        self.check_load()

        n_train_samples = self.n_train_samples
        n_test_samples  = self.n_test_samples

        data = self.data[n_train_samples : n_train_samples+n_test_samples, : ]
        return data

    def get_features(self):
        self.check_load()
        return self.features

    # legacy
    def discretize(self, nbins = 10, method = "cluster"):
        self.check_load()

        features = self.features
        data = self.data

        print "discretizing..."
        for i in range(len(features)):
            f = features[i]
            if f.type is not FEATURE_CONTINUOUS:
                continue

            print "discretizing f %d"%i
            values = np.take(data, i, axis = 1)
            values = f.discretize( values, nbins, method)
            data[:,i] = values

        self.is_discretized = True
        return self

    def _export_data(self, data_file, data):
    
        of = open(data_file, 'w')
        
        for row in data:
            string = ""
            for v in row:
                string += "{0} ".format(v) 
                #of.write("{0} ".format(v))
            string = string.rstrip(" ")
            of.write(string+"\n")
        of.close()


    def export(self, feature_file, train_data_file, test_data_file):
        self.check_load()

        print "exporting..."
        features = self.features

        print "exporting to feature file..."
        of = open(feature_file, 'w')
        for f in features:
            of.write(str(f))
            of.write("\n")
        of.close()

        train_data = self.get_train()
        print "exporting to train data file...",
        t1 = time()
        np.save(train_data_file, train_data)
        t2 = time()
        print "%.2f"%(t2-t1)

        test_data  = self.get_test()
        print "exporting to test data file...",
        t1 = time()
        np.save(test_data_file,  test_data)
        t2 = time()
        print "%.2f"%(t2-t1)

        return self

    def load_arff(self, arff_file, train_to_test = "9:1"):

        data, meta = arff.loadarff(arff_file)
        names = meta.names()
        types = meta.types()

        features = []
        for name in names:
            feature = Feature()
            feature.name = name
            
            type_, values_ = meta[name] 
            if type_ is "numeric":
                feature.type = FEATURE_CONTINUOUS
                
            elif type_ is "nominal":
                feature.type = FEATURE_DISCRET
                values = []
                indices = {}
                for v in values_:
                    indices[v] = len(values)
                    values.append(v)

                feature.indices = indices
                feature.values = values
                feature.n_values = len(values)

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

        n_samples = new_data.shape[0]

        if train_to_test == "9:1":

            self.n_train_samples = 9 * (n_samples // 10)
            self.n_test_samples  = n_samples - self.n_train_samples
            print "train %d, test %d"%(self.n_train_samples, self.n_test_samples)

        return self

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

    def _load_data(self, features, data_file, sep=" "):
        ''' load data by using loadtxt() shipped by numpy,
            the dataset should by 2-dimension array, separated by space(' ') 
       
            the number of columns should equals to the number of features in feature_file
        '''
        #data = np.loadtxt( data_file )
        #if data.ndim != 2:
        #    raise Exception, "data is not 2-dimension array"

        names = [ f.name for f in features]
        d = pd.read_csv( data_file, names=names, sep=sep, header=None) # separators XXX
    
        data = np.array(d)
        #print data

        n_samples, n_features = data.shape[0], data.shape[1]
        if n_features != len(features):
            raise Exception, "number of features {0} in data is not consistent with that {1} in feature file".format(n_features, len(features))

        return data

    def load_raw(self, feature_file, train_data_file, test_data_file, sep=" "):
        ''' load from non-arff raw file '''

        print "load feature file..."
        features   = self._load_features( feature_file )

        print "load train data...",
        t1 = time()
        if train_data_file.endswith(".npy"):
            train_data = np.load(train_data_file)
        else:
            train_data = self._load_data(features, train_data_file)
            
        t2 = time()
        print "%.2f"%(t2-t1)
        
        n_train_samples = train_data.shape[0]
        n_features      = train_data.shape[1]

        if test_data_file is not None:
            
            print "load test data...",
            t1 = time()
            if test_data_file.endswith(".npy"):
                test_data = np.load(test_data_file)
            else:
                test_data = self._load_data(features, test_data_file)

            t2 = time()
            print "%.2f"%(t2-t1)
        
            n_test_samples  = test_data.shape[0]

            print "concatenate...",
            t1 = time()
            data = np.concatenate((train_data, test_data), axis=0)
            t2 = time()
            print "%.2f"%(t2-t1)
        else:
            data = train_data

        #print data

        td0 = time()  
        print "parsing from data...",
        for i in range(n_features):
            f = features[i]
            t0 = time()

            values = data[:, i]

            f.parse_meta_from_data( values )

            if f.type is FEATURE_CONTINUOUS and self.is_discretize:
                t1 = time()

                values = values.astype(float) #XXX
                #print values

                f.discretize( values, self.nbins, self.dmethod)
                t2 = time()
                print "discretize takes  %.2fs"%(t2-t1)

            if f.type is FEATURE_DISCRET:
                values =  f.transfer( values )
                data[:,i] = values

            t3 = time()
            #print "[%2d] %25s, t=%d %.2fs"%(i, f.name, f.type, t0-t3)

        td1 = time()
        print "%.2fs"%(td1-td0)

        data = data.astype("float") # for dataset read from cvs

        if test_data_file is not None:
            self.n_test_samples = n_test_samples
        else :
            self.n_test_samples = 0

        self.n_train_samples = n_train_samples
        self.data = data
        self.features = features
        return self

    def load_existed(self, feature_file, train_data_file, test_data_file):
        ''' load from preprocessed data, 
            for discret feature, values are all transferred to numbers
            for continous feature, values are all float type

            * discretize part has already finished, 
            DO NOT call Preprocessor.discretize() after this function '''

        features = self._load_features( feature_file )

        print "load train data...",
        t1 = time()
        train_data = np.load(train_data_file)
        t2 = time()
        print "%.2f"%(t2-t1)
        
        n_train_samples = train_data.shape[0]
        n_features      = train_data.shape[1]

        print "load test data...",
        t1 = time()
        test_data  = np.load(test_data_file)
        t2 = time()
        print "%.2f"%(t2-t1)

        n_test_samples  = test_data.shape[0]
        data = np.concatenate((train_data, test_data), axis=0)

        #data = data.astype(float)

        self.n_train_samples = n_train_samples
        self.n_test_samples  = n_test_samples
        self.data   = data
        self.features = features
        return self

    def load(self, feature_file, train_data_file=None, test_data_file=None, sep=" ", is_discretize = True, nbins=10, dmethod="cluster"):
        ''' load from raw data format, the data need to be preprocessed '''

        self.nbins = nbins
        self.dmethod = dmethod
        self.is_discretize = is_discretize

        if feature_file.endswith(".arff"):
            self.load_arff(feature_file)
        else:
            self.load_raw(feature_file, train_data_file, test_data_file, sep=sep)
        return self
    
    def clean(self, data_in, sep=" "):

        data_out = data_in+".npy"

        t1 = time()
        data = np.loadtxt( data_in )
        t2 = time()
        print "loaded data... %.2fs"%(t2-t1)

        # remove the first col
        t1 = time()
        data = np.delete( data, 0, axis=1)
        t2 = time()
        print "finished removing phone number... %.2fs"%(t2-t1)

        print "data shape", data.shape

        def remove_missing_value( X, strategy="most_frequent", missing_values="NaN"):

            if strategy is "remove":
                X = X[ ~np.isnan(X).any(axis=1) ]
                return X

            imp = Imputer(missing_values='NaN', strategy=strategy, axis=0)
            X = imp.fit_transform(X)
            return X

        t1 = time()
        data = remove_missing_value( data, strategy = "mean")
        t2 = time()
        print "remove missing value... %.2fs"%(t2-t1)

        print "n_samples %d n_features %d"%data.shape

        t1 = time()
        np.save(data_out, data )
        t2 = time()
        print "save... %.2fs"%(t2-t1)

        return data


