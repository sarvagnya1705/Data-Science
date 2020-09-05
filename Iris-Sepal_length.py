#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
# Import Dataset from sklearn 
from sklearn.datasets import load_iris
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


iris=load_iris()
iris


# In[3]:


df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df


# In[4]:


dff=pd.DataFrame(data=iris.target,columns=['species'])
dff
#reg=linear_model.LinearRegression()


# In[5]:


def converter(specie):
    if specie==0:
        return 'setosa'
    elif specie==1:
        return 'versicolor'
    else:
        return 'virginica'
dff['species']=dff['species'].apply(converter)
df=pd.concat([df,dff],axis=1)


# In[8]:


df.info()


# In[9]:


df


# In[10]:


sns.pairplot(df, hue= 'species')


# In[13]:


#for calculating the error of the model,we'll predict sepal length 
#then compare it with the actual sepal length given in our dataset

#linear reg object
reg=linear_model.LinearRegression()
#species->number dtype
df.drop('species',axis=1,inplace=True)
dff=pd.DataFrame(columns=['species'],data=iris.target)
df=pd.concat([df,dff],axis=1)
df


# In[19]:


y=df['sepal length (cm)']
x=df.drop(axis=1,labels='sepal length (cm)')
y


# In[20]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.33,random_state=101)


# In[21]:


reg.fit(x_train,y_train)


# In[26]:


reg.predict(x_test)
predicted=reg.predict(x_test)


# In[27]:


print('Mean Absolute Error ', mean_absolute_error(y_test, predicted))
print('Mean Squared Error ', mean_squared_error(y_test, predicted))
print('Mean Root Squared Error ', np.sqrt(mean_squared_error(y_test, predicted)))
