import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression

class hdlr2(x_data, y_data):

	x_data=x_data
	y_data=y_data

	def fit(x_data, y_data):
		return LogisticRegression(*args, solver='lbfgs')
	
	def get_params(x_data, y_data):
		return LogisticRegression.fit(X, y).coef_