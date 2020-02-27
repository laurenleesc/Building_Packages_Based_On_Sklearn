import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
from high_dim_log_reg import Hdlr

#from high_dim_log_reg.datasets import bernoulli

X=np.load('high_dim_log_reg/datasets/bernoulli_X.npy')
b=np.load('high_dim_log_reg/datasets/bernoulli_b.npy')
y=np.load('high_dim_log_reg/datasets/bernoulli_y.npy')
means=np.load('high_dim_log_reg/datasets/bernoulli_means.npy')

#first with statsmodels package
model = Logit(y, X)
 
result = model.fit()

est_betas = result.params

#next with sklearn
model2 = LogisticRegression()
result2 = model2.fit(X, y)

est_betas2 = result2.coef_

#print(est_betas2)

#Lastly, with our new function, no bias correction yet 

model3 = Hdlr()
result3 = model3.fit(X, y)

est_betas3 = result3.coef_

print(est_betas3)
