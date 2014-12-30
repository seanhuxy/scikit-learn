from preprocessor import Preprocessor

feature_in = "feature.in"
data_in = "data.in"

feature_out = "feature.out"
data_out = "data.out"

preprocessor = Preprocessor()
preprocessor.load_raw( feature_in,  data_in)
preprocessor.transfer()
preprocessor.export( feature_out, data_out)

X = preprocessor.get_X()
y = preprocessor.get_y()

print X
print y
