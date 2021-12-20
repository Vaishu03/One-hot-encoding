#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd


# In[30]:


df = pd.read_csv(r"E:\ML\carprices.csv")
df


# In[31]:


dum = pd.get_dummies(df.Cars)
dum


# In[32]:


merge = pd.concat([df,dum],axis = 'columns')
merge


# In[33]:


final = merge.drop(['Cars','Toyota'],axis = 'columns')
final


# In[48]:


from sklearn.linear_model import LinearRegression
obj = LinearRegression()


# In[38]:


X = final.drop(['Price'],axis = 'columns')
X


# In[39]:


y = final.Price
y


# In[50]:


obj.fit(X,y)


# In[54]:


#price of a mercedez benz i.e. 4 yr old with mileage 45000
obj.predict([['45000','4','0','0','1']])


# In[56]:


#price of a BMW X5 i.e. 7 yr old with mileage 86000
obj.predict([['86000','7','0','1','0']])


# In[58]:


#predicting the accuracy of the model
obj.score(X,y)

