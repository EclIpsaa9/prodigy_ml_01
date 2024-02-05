#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.gridspec import GridSpec
import csv


# In[11]:


df = pd.read_csv("C:/Users/DANICA/Desktop/prodigyHousing.csv")


# In[12]:


print(df.columns)


# In[14]:


pd.options.display.max_columns=None 
df.info()


# In[29]:


Category_features = ['mainroad','guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'] 
Numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

for feature in df.columns: 
    print(feature, " : ", df[feature].unique())
    fig, axes = plt.subplots(2, 4, figsize=(20,10))
    fig, axes = plt.subplots(len(Category_features)//4 + 1, 4, figsize=(16, 4 * (len(Category_features)//4 + 1)))
    
for k in range(len(Category_features)):
    num = []
    for t in df[Category_features[k]].unique():
        num.append(df[Category_features[k]].tolist().count(t))
    
    row, col = k // 4, k % 4
    axes[row][col].pie(num, labels=df[Category_features[k]].unique(), autopct="%.2f%%", labeldistance=1.15,
                       wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                       textprops={'color': 'lightgreen', 'fontsize': 15}, colors=sns.color_palette('Blues_r'))
    axes[row][col].set_title(Category_features[k])


for i in range(len(Category_features), len(Category_features)//4 + 1 * 4):
    row, col = i // 4, i % 4
    axes[row][col].axis('off')

plt.tight_layout()
plt.show()


# In[42]:


import random
random_colors = ["#" + ''.join([random.choice('0123456789ABCDEF')for j in range(6)]) for i in range(5)]

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(nrows=2, ncols=3)

for k in range(len(Numeric_features)-1):
    ax = fig.add_subplot(gs[k//3, k%3])
    sns.countplot(ax=ax, data=df, x=Numeric_features[k+1], palette=sns.color_palette('pastel'))

    k += 1
    sns.countplot(ax=fig.add_subplot(gs[k//3, k%3:]), data=df, x=Numeric_features[0], palette=sns.color_palette('pastel'))

plt.tight_layout()
plt.show()


# In[43]:


#histograms
import random
random_colors = ["#" + ''.join([random.choice('0123456789ABCDEF')for j in range(6)]) for i in range(5)]

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(nrows=2, ncols=3)

for k in range(len(Numeric_features)-1):
    ax = fig.add_subplot(gs[k//3, k%3])
    sns.histplot(ax=ax, data=df, x=Numeric_features[k+1], discrete=True, stat="percent")

    k += 1
    ax0 = fig.add_subplot(gs[k//3, k%3:])
    sns.histplot(ax=ax0, data=df, x=Numeric_features[0], discrete=True, stat="percent")
    ax0.set_ylim(0, 1.75)

plt.tight_layout()
plt.show()


# In[44]:


#for boxplots
sns.set_style("darkgrid")

# Single boxplot
plt.figure(figsize=(5, 8))
plt.boxplot(x=df['price'], notch=True)
plt.ylabel('price')
plt.show()

# Multiple boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 7))
sns.boxplot(ax=axes[0], data=df, y="price")
sns.boxplot(ax=axes[1], data=df, y="price", showcaps=False, whiskerprops={"linestyle": 'dashed', "lw":4})
sns.boxplot(ax=axes[2], data=df, y="price", notch=True, showmeans=True, meanline=True,
            meanprops={"color": "r", "lw":2}, medianprops={"color": "c", "lw":3})
plt.tight_layout()
plt.show()


# In[45]:


print(df.shape)
r = 3
z_score = (df['area'] - df['area'].mean()) / df['area'].std()
df = df[(z_score > (-1) * r) & (z_score < r)]
print(df.shape)


# In[46]:


r = 3
lower_limit = df['area'].mean() - r * df['area'].std()
upper_limit = df['area'].mean() + r * df['area'].std()
print(df.shape)
df = df[(df['area'] >= lower_limit) & (df['area'] <= upper_limit)]
print(df.shape)


# In[51]:


import numpy as np 
import pandas as pd

ex_df = pd.DataFrame({
    'feature_name': ['square', np.nan, 'oval', 'square', 'circle', np.nan, 'triangle'],
    'feature_name2': [1, np.nan, 3, 4, 5, np.nan, 7],
    'feature_name3': ['squares', 'triangles', np.nan, 'circles', 'ovals', np.nan, 'squares'],
})

# Checking for missing values
print(ex_df)
print(ex_df.isnull().sum())


# In[52]:


# Defining regression models
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor 
from sklearn.svm import SVR
Models = [
    ["Linear Regression", LinearRegression()],
    ["Decision Tree Regressor", DecisionTreeRegressor()],
    ["RandomForestRegressor", RandomForestRegressor()],
    ["Gradient Boosting Regressor", GradientBoostingRegressor()],
]
print(ex_df)


# In[ ]:




