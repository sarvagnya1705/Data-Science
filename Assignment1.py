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


# In[4]:


df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df


# In[6]:


dff=pd.DataFrame(data=iris.target,columns=['species'])
dff
#reg=linear_model.LinearRegression()


# In[7]:


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


# In[12]:


sns.pairplot(df, hue= 'species')


# In[22]:


#for calculating the error of the model,we'll predict sepal length 
#then compare it with the actual sepal length given in our dataset

#linear reg object
reg=linear_model.LinearRegression()
#species->number dtype
df.drop('species',axis=1,inplace=True)
dff=pd.DataFrame(columns=['species'],data=iris.target)
df=pd.concat([df,dff],axis=1)
df


# In[49]:


y=df['sepal length (cm)']
z=df['sepal width (cm)']
x=df.drop(axis=1,labels=['sepal length (cm)', 'sepal width (cm)'])
x


# In[58]:


x_train, x_test, y_train, y_test = train_test_split(y,z, test_size=0.2,random_state=101)


# In[83]:


plt.scatter(x_train,y_train)
plt.title("LENGTH VS WIDTH")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")


# In[61]:


x=x_train
y=y_train
mean_x=x.mean()
mean_y=y.mean()
#calculating B1 and B0
b1=np.divide(np.sum(np.multiply(np.subtract(x,mean_x),np.subtract(y,mean_y))),np.sum(np.square(np.subtract(x,mean_x))))
print("B1: ",b1)
b0=np.subtract(mean_y,np.multiply(b1,mean_x))
print("B0: ",b0)


# In[63]:


#Y=b0+b1*X
prediction=b0+b1*x_test
prediction
#prediction will be compared to y_test for error


# In[67]:


rmse=np.sqrt(np.divide(np.sum(np.square(np.subtract(prediction,y_test))),x.size))


# In[70]:


print("Root mean squared error is for sepal length vs sepal width is: ",rmse)


# NON LINEAR CURVES
# 

# In[80]:


prediction=b0+b1*np.square(x_test)


# In[77]:


rmse=np.sqrt(np.divide(np.sum(np.square(np.subtract(prediction,y_test))),x.size))


# In[78]:


print("Root mean squared error is for sepal length vs sepal width is using non linear curve: ",rmse)
