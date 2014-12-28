from loadtext import Preprocessor

feature_file = "feature_input.in"
data_file = "data_input.in"

feature_output = "feature.out"
data_output = "data.out"

preprocessor = Preprocessor()
preprocessor.load( data_file, feature_file)
preprocessor.export( data_output, feature_output)  
