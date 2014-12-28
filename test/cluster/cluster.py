import numpy as np
from numpy import array
from scipy.cluster.vq import vq, kmeans2



data = array([1,1,2,2,3,3,4,4,5,5])
k = 5

code_book, _ = kmeans2( data, k, iter=20, minit="random")

code_book = np.sort(code_book)

#newdata = array([0.2, 0.5, 1.2, 3.6, 90])
newdata = np.float(0.1)
code, dists = vq( newdata, code_book)

print code_book
print code


