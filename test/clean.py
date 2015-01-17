import sys
import numpy as np
from time import time
from preprocessor import Preprocessor

cwd = os.getcwd()
path = cwd+"/dataset/"


if __name__ == "__main__":

    print sys.argv
    fname = sys.argv[1]
    print fname

    p = Preprocessor()
    p.clean(path+fname)






