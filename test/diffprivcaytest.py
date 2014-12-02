
import numpy as np
from sklearn.datasets import load_iris
#from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DiffPrivacyDecisionTreeClassifier

from sklearn.datasets import load_svmlight_file

filename = "../dataset/adulta3a.txt"

X, y = load_svmlight_file(filename)
X = X.toarray()

budget = 1.0
max_depth = 5

clf = DiffPrivacyDecisionTreeClassifier(random_state=2, 
										diffprivacy_mech=2, 
										budget = budget,
										max_depth = max_depth)
exp_output = cross_val_score(clf, X, y, cv=10)

clf.diffprivacy_mech=1
lap_output = cross_val_score(clf, X, y, cv=10)

clf.diffprivacy_mech=0
no_output =  cross_val_score(clf, X, y, cv=10)


print "============================================="
print "dataset: ", filename
print "# of samples:\t",  X.shape[0]
print "# of features:\t", X.shape[1]
print "# of outputs:\t",  y.shape[0]

print "Budget =",budget,"; max_depth =",max_depth

print "No Diffprivacy   :\t", np.average(no_output)
print "Laplace Mech     :\t", np.average(lap_output)
print "Exponential Mech :\t", np.average(exp_output)
