#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'notebook')

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
warnings.filterwarnings(action="ignore", module="sklearn", message="^Objective did not")


# In[15]:
# First we are going to show, why a linear regression with dim(X) = 2 can't produce good result for a non-linear function such as XOR.

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])


# In[23]:


plt.close()

plt.scatter(X[:, 0], y)
plt.gca().set_xlim(-1, 2)
plt.gca().set_ylim(-1, 2)


# In[24]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)


# In[25]:


model.coef_, model.intercept_

# Out[25]: (array([ 0.00000000e+00, -2.22044605e-16]), 0.5000000000000001)
# This is obviously just a straight line parallel to the X axis.

# Now, we're going to add an extra dimension to X and try training the neural network.

# In[26]:


boundaryLow = -1
boundaryHigh = 2
interval = np.linspace(boundaryLow, boundaryHigh)
result = interval * model.coef_[0] + model.intercept_

plt.close()
plt.scatter(X[:, 0], y)
plt.plot(interval, result)
plt.gca().set_xlim(boundaryLow, boundaryHigh)
plt.gca().set_ylim(boundaryLow, boundaryHigh)


# In[62]:


X=np.insert(X, 1, 0, 1)
for i in range(len(X[:,1])):
    X[i, 1] = X[i,0]*X[i,2]
X


# In[63]:


model = LinearRegression()
model.fit(X, y)
model.coef_, model.intercept_

# Out[63]: (array([ 1., -2.,  1.]), 1.6653345369377348e-16)

# In[ ]:




