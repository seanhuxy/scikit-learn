
# replace ^A by space

# remove the first col
def remove_phone_num(X):
    data = np.delete( X, 0, axis=1)
    return data

#data = remove_col( data, 0)
#del features[0]


# missing value
from sklearn.preprocessing import Imputer


def remove_missing_value( X, strategy="remove", missing_values="NaN"):

    if missing_values is "NaN" or np.isnan(missing_values):
        missing_values = np.NaN

    if strategy is "remove":
        del_row = []
        for i, row in enumerate(X):
            for col in row:
                if col is missing_values:
                    del_row.append(i)

        array = np.delete( array, del_row, axis=0)
        return array

    imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
    X = imp.fit_transform(X)
    return X



 no diffprivacy
 1.   not discretized
 2.   discretized

 laplace 
 3.   discretized

 exponential
 4.   not discretized
 5.   discretized

# criterion
    entropy, gini

# budget
budget = [0.5, 1.0 , 3.0, 5.0, 7.0, 10.0]

# features: 70
    sorted 

n_features = [  70, 60, 50, 40, 30, 20, 10,
                20, 17, 14, 11, 9, 7, 5, ]

# samples: 200,000
    2,000,000; 
    1,500,000; 
    1,000,000; 
      500,000; 
      100,000;
       50,000; 
       10,000;

n_samples  = [  2000000,
                1500000,
                1000000,
                 500000,
                 100000,
                  50000,
                  10000 ]

def no_diffprivacy_test( X, y ):
    diffprivacy_mech = "no"

    pass

def laplace_test():
    pass

def exponential_test():
    pass 

# forest



