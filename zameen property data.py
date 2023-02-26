#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#import errors and r square
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import scipy.stats as stats
import seaborn as sns


# In[2]:


df=pd.read_csv('Zameen_data.csv')
df.head()


# In[32]:


houses=df.shape[0]
houses


# In[33]:


houses_in_isb = df[df['city'] == 'Islamabad'].shape[0]
print("Total Number of Houses in city Islamabad:", houses_in_isb)
houses_in_khi = df[df['city'] == 'Karachi'].shape[0]
print("Total Number of Houses in city Karachi:", houses_in_khi)


# In[35]:


#probability of picking house in Islamabad 
probability = (houses_in_isb/houses)*100
print('Probability of getting a house is Islamabad City: {0:.2f}'.format(probability )+'%')


# In[37]:


#probability of picking house in Karachi
probability = (houses_in_khi/houses)*100
print('Probability of getting a house is Karachi City: {0:.2f}'.format(probability )+'%')


# In[40]:


cond_prob = (houses_in_khi/houses) * ((houses_in_khi - 1)/(houses - 1)) 
print("The Probability of getting a house in Karachi and again picking a house from Karachi is {0:.9f}".
      format(cond_prob*100)+'%')


# In[50]:


plt.rcParams['figure.figsize'] = (8,4)
plt.xticks(rotation=27)
sns.distplot(df['price'])
plt.title('Distribution of Prices')
plt.show()


# In[43]:


df.isnull().sum()


# In[8]:


NAN_value=pd.isnull(df).sum()
NAN_value[NAN_value>0]


# In[13]:


plt.figure(figsize=(5,5))
sns.heatmap(df.isnull().T);


# In[14]:


#sum of missing values and show values and their percentages with transpose dataframe
pd.set_option('display.max_rows', None)
missing_value= pd.DataFrame(df.isnull().sum().sort_values(ascending = False), columns = ['sum_of_missing_values'])
missing_value['percentage_of_missing_values'] = missing_value['sum_of_missing_values']/df.shape[0]*100
missing_value = missing_value[missing_value['sum_of_missing_values']>0]
missing_value


# In[15]:


#display count plot to check missing values
plt.figure(figsize = (10, 5))
sns.barplot(x = missing_value.index, y = missing_value['sum_of_missing_values'],palette ='Blues',edgecolor='black')
plt.xticks(rotation =70)
plt.title("Bar plot for checking missing values")
plt.xlabel("Features")
plt.ylabel("Total Missing Values")
plt.show()


# In[16]:


df.drop(['agent'], axis=1, inplace=True)
df.drop(['agency'], axis=1, inplace=True)


# In[18]:


df.isnull().sum().any()


# In[22]:


plt.figure(figsize=(5,5))
sns.heatmap(df.isnull().T);


# In[24]:


df_object = df.select_dtypes(include=['object'])
df_object.head()


# In[25]:


df_exclude_object = df.select_dtypes(exclude=['object'])
df_exclude_object.head()


# In[51]:


np.random.seed(6)


# In[52]:


# lets take 500 sample values from the dataset of 168446 values
sample_ages = np.random.choice(a= df['price'], size=6500)

# getting the sample mean
print ("Sample mean:", sample_ages.mean() )          

# getting the population mean
print("Population mean:", df['price'].mean())


# In[53]:


#Are house prices in karachi different from the House Prices of Other cities?
# lets import z test from statsmodels
from statsmodels.stats.weightstats import ztest

z_statistic, p_value = ztest(x1 = df[df['city'] == 'Karachi']['price'],
                             value = df['price'].mean())

# lets print the Results
print('Z-statistic is :{}'.format(z_statistic))
print('P-value is :{:.50f}'.format(p_value))


# In[ ]:


#If the P value if less than 0.05, then we can reject our null hypothesis against the alternate hypothesis.


# In[ ]:


#Now, let's also see if house prices in Stone Brook neighborhood are different from the houses in the rest of the neighborhoods.


# In[ ]:




