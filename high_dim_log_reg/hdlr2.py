import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression

class hdlr2(X, y):

	def fit(X, y):
		return LogisticRegression(*args, solver='lbfgs')
	
	def get_params(X, y):
		return LogisticRegression.fit(self, X, y).coef_