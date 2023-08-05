#!/usr/bin/env python
# coding: utf-8

# # IMPORTING REQUIRED LIBRARIES

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# # reading data
# 

# In[4]:


df1 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")
df2=pd.read_csv("mental-and-substance-use-as-share-of-disease.csv")


# In[5]:


df1.head()


# In[6]:


df2.head()


# # merging two dataset

# In[7]:


data=pd.merge(df1,df2)
data.head()


# # DATA CLEANING

# In[8]:


data.isnull().sum()


# In[9]:


data.drop('Code',axis=1,inplace=True)
data.head()


# In[10]:


data.size,data.shape


# In[11]:


data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns', inplace=True)


# In[12]:


data.head()


# # EXPLORATORY DATA ANALYSIS

# In[19]:


plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,cmap='Purples')
plt.plot()


# In[23]:


sns.jointplot(x='Schizophrenia',y='mental_fitness',data=data,kind='hex',color=None)
plt.show()


# In[25]:


sns.jointplot(x='Bipolar_disorder',y='mental_fitness',data=data,kind='kde',color='m')
plt.show()


# In[26]:


sns.pairplot(data,corner=True)
plt.show()


# In[27]:


mean=data['mental_fitness'].mean()
mean


# In[29]:


fig=px.pie(data,values='mental_fitness',names='Year')
fig.show()


# In[30]:


fig=px.bar(data.head(10),x='Year',y='mental_fitness',color='Year',template='ggplot2')
fig.show()


# # YEARWISE VARIATIONS IN MENTAL FITNESS OF DIFFERENT COUNTRIES
# 
# 

# In[31]:


fig = px.line(data, x="Year", y="mental_fitness", color='Country',markers=True,color_discrete_sequence=['red','blue'],template='plotly_dark')
fig.show()


# In[32]:


df=data.copy()
df.head()


# In[33]:


df.info()


# In[34]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.columns:
    if df[i].dtype == 'object':
        df[i]=l.fit_transform(df[i])


# In[36]:


X = df.drop('mental_fitness',axis=1)
y = df['mental_fitness']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)


# In[37]:


X = df.drop('mental_fitness',axis=1)
y = df['mental_fitness']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)


# # support vector regression

# In[38]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

svr = SVR()
svr.fit(xtrain, ytrain)

# model evaluation for training set
ytrain_pred = svr.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = np.sqrt(mean_squared_error(ytrain, ytrain_pred))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for the training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = svr.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for the testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# # Gradient Boosting Regressor:

# In[39]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

gbr = GradientBoostingRegressor()
gbr.fit(xtrain, ytrain)

# model evaluation for training set
ytrain_pred = gbr.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = np.sqrt(mean_squared_error(ytrain, ytrain_pred))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for the training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
ytest_pred = gbr.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))
r2 = r2_score(ytest, ytest_pred)

print("The model performance for the testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[ ]:




