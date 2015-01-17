import sys
import os
import numpy as np

cwd = os.getcwd()
path = cwd+"/dataset/"
sys.path.append(cwd)


from preprocessor import Preprocessor



if __name__ == "__main__":

    fname = "adult_nomissing.arff"
    print sys.argv
    #fname = sys.argv[1]
    print fname

    p = Preprocessor()
    p.load_arff(path+fname)

    p.export(path+"feature.in", path+"data.npy", path+"test.npy",)




