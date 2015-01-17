


feature_importances = []


def filte_feature( data, n_features ):

	#data = np.take( data, feature_importances[:n_features]))
	data = data[ : , feature_importances[ :n_features] ]
	return data

def filte_sample( data, n_samples ):
	
	data = data[ : n_samples]
	return data
	# or bagging

	


def main( data,  is_discretized )

	#get discretized data
	#is_discretized = False

	tree = NBTree()

	tree.diffprivacy_mech = "no"
	
	for n_f in n_features:
		data_f = filte_feature( data, n_f)

		for n_s in n_samples:
			data_s = filte_sample( data_f, n_s)
			

				tree = NBTree( )






