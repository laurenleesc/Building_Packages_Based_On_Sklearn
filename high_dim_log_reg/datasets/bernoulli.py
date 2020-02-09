import numpy as np

def load_bernoulli_data():
	X=np.load('bernoulli_X.npy')
	b=np.load('bernoulli_b.npy')
	y=np.load('bernoulli_y.npy')
	means=np.load('bernoulli_means.npy')
	return X, b, y, means
	#print('You now have numpy arrays X, b, y and means.')
