from sklearn.preprocessing import OneHotEncoder
import numpy as np 

data = np.genfromtxt("../dataset/adult.data", delimiter=",")

print data[0]

enc= OneHotEncoder(data)
print enc
print enc.n_values_

