import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
#from high_dim_log_reg import hdlr2

#from high_dim_log_reg.datasets import bernoulli

X=np.load('high_dim_log_reg/datasets/bernoulli_X.npy')
b=np.load('high_dim_log_reg/datasets/bernoulli_b.npy')

max_=round(max(b),2)

print('Our current dataset has the highest beta/signal of', max_, '. What would happen if we were to scale our data prior to our MLE?')