
# coding: utf-8

# In[59]:

import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
import pca


# Quiz 1 - Question number from part a to c

# In[24]:

data = genfromtxt('dataset_1.csv', delimiter=',', skip_header=1)


# In[25]:

x = data[0:,0]


# In[52]:

y = data[0:,1]


# In[27]:

z = data[0:,2]


# In[33]:

x_var = np.var(x)


# In[35]:

x_var


# In[36]:

y_var = np.var(y)


# In[37]:

y_var


# In[38]:

z_var = np.var(z)


# In[39]:

z_var


# In[46]:

covar_xy = np.cov(x,y)


# In[48]:

covar_xy


# In[49]:

covar_yz = np.cov(y,z)


# In[50]:

covar_yz


# In[54]:

pca_mat = pca.PCA(data)


# In[55]:

pca_mat


# Quiz 1 - Question 3 part 2

# In[57]:

A = np.array([[0,-1],[2,3]])


# In[58]:

A


# Calculating Eigen Values and Eigen Vectcors 

# In[60]:

eig_val, eig_vec = LA.eig(A)


# In[61]:

eig_val


# In[62]:

eig_vec


# In[ ]:



