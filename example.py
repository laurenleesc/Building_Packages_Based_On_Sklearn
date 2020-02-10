import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import Logit

#from high_dim_log_reg.datasets import bernoulli

X=np.load('high_dim_log_reg/datasets/bernoulli_X.npy')
b=np.load('high_dim_log_reg/datasets/bernoulli_b.npy')
y=np.load('high_dim_log_reg/datasets/bernoulli_y.npy')
means=np.load('high_dim_log_reg/datasets/bernoulli_means.npy')

model = Logit(y, X)
 
result = model.fit()

tb = np.squeeze(b)
est_betas_ub = result.params + 2*result.bse
est_betas_lb = result.params - 2*result.bse
captured = np.where((tb<=est_betas_ub)&(tb>=est_betas_lb),1,0)
print("Proportion of Betas within 2SE of True Value: "+str(sum(captured)/len(captured)))
print("")
diff = (result.params - np.squeeze(b))
print("Average |Diff| minus |SE|:                    "+str(np.mean(abs(diff)-abs(result.bse))))


n=len(X)
p=len(b)
mu = 0
stdev = 1.0

plt.scatter(np.squeeze(b),result.params, label="Estimated Betas vs. True Betas")
plt.errorbar(np.squeeze(b),result.params,yerr=result.bse, fmt='o')
plt.scatter(np.squeeze(b),np.squeeze(b), label="True Betas vs. True Betas")
#plt.ylim(-7.03,7.03)
#plt.xlim(-7.03,7.03)
plt.xlabel('True Beta Value')
plt.ylabel('Beta Value')
plt.title('Beta-Beta Plot\nn='+str(n)+', p='+str(p)+', beta_mu='+str(mu)+', beta_stdev='+str(stdev)+'\nProportion Captured: '+str(sum(captured)/len(captured)))
plt.legend()
plt.show()
