#!/usr/bin/env python
# coding: utf-8

# <a href='https://www.hexnbit.com/'> <img src='https://www.hexnbit.com/wp-content/uploads/2019/09/hexnbit_final_66px.png'/> </a>

# All cells must be suitably commented / documented.

# ### Read Dataset

# In[39]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


df=pd.read_csv("glass.csv")
df.head()


# In[41]:


#creating a new row
new_row = pd.DataFrame({'1.52101':1.52101, '13.64':13.64, '4.49':4.49,'1.1':1.10,
                        '71.78':71.78, '0.06':0.06, '8.75':8.75,
                        '0':0.00, '0.1':0.00, '1':1},index =[0])

#adding the new row to the dataframe
df1 = pd.concat([new_row, df]).reset_index(drop = True)

#renaming the columns
df1.columns = ['RI: refractive index','Na: Sodium','Mg: Magnesium','Al: Aluminum','Si: Silicon','K: Potassium','Ca: Calcium','Ba: Barium','Fe: Iron','Type of glass']
df1.head()


# ### Check for Missing Data

# In[42]:


#cheking missing data using heatmap

plt.figure(figsize=(12,10))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='plasma')


# There are no missing values beacuse if there were any missing values we would have gotten yellow line

# In[43]:


#exploratory data analysis

#representing all the correlation between all the features visually

sns.heatmap(df1.corr(),annot=True,cmap="Blues")


# we can see that Al,Na and Ba are stronglyn correlated to the type of glass

# In[44]:


#exploratory Data analysis

#plotting bar graphs of count of all glass types

sns.countplot(x='Type of glass',data=df1)


# it is clear that glass types 1 and 2 contain more than 65% of the total data

# In[45]:


#create box plots for all the features


fig,ax = plt.subplots(ncols=6,nrows=2,figsize=(20,10))
index=0
ax=ax.flatten()

for col, value in df1.items():
    if col!='type':
        sns.boxplot(y=col,data=df1,ax=ax[index])
        index+=1
    


# ### Handle Categorical Values

# The categorical variables in the given data set have already been handled

# ### Split Data for Training and Testing

# In[46]:


from sklearn.model_selection import train_test_split 


# In[47]:


#Splitting data into training(70% of data) and testing data(30% of the data)

X_train, X_test, y_train, y_test = train_test_split(df1.drop('Type of glass',axis=1), 
                                                    df1['Type of glass'], test_size=0.30, 
                                                    random_state=101)


# ### Apply different Classification Algorithms and tune them

# In[48]:


# support vector machine

from sklearn.svm import SVC #importing the support vector machine model 
model=SVC() #initializing the support vector machine  model
model.fit(X_train,y_train) # training the model
predictions=model.predict(X_test) # predicting the glass types for test data


# In[49]:


#Logistic Regression

from sklearn.linear_model import LogisticRegression #importing the logistic regression model
logmodel = LogisticRegression(max_iter=800) #initializing the logistic regression  model
logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train,y_train) # training the  model
predictions1 = logmodel.predict(X_test) # predicting the glass types for test data


# In[50]:


#Descision Trees

from sklearn.tree import DecisionTreeClassifier # importing the descision trees model
model1 = DecisionTreeClassifier() # initializing the descision trees model
model1.fit(X_train, y_train) #traing the model
predictions2=model1.predict(X_test) # predicting the glass types for test data


# In[51]:


#K nearest neighbours 

from sklearn.neighbors import KNeighborsClassifier # importing the K nearest neighbors model
classifier = KNeighborsClassifier(n_neighbors=5) # initializing the K nearest neighbors model
classifier.fit(X_train, y_train)# training the model
y_pred = classifier.predict(X_test) # predicting the glass types for test data


# In[52]:


# Hyper peramter tuning in support Vector machine 

from sklearn.model_selection import GridSearchCV
grid_parameters = {'C': [0.001,0.1,1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]} 
grid = GridSearchCV(SVC(),grid_parameters,verbose=1)
grid.fit(X_train,y_train)# training the model after tuning
grid_predictions = grid.predict(X_test) # predicting the glass types for test data after tuning


# In[53]:


# Hyper peramter tuning in Logistic regression

from sklearn.model_selection import GridSearchCV
params = {'penalty' : ['l1', 'l2'],'C' : np.logspace(-4, 4, 20),'solver' : ['liblinear']}
grid1 = GridSearchCV(LogisticRegression(),params,verbose=1)
grid1.fit(X_train,y_train)#traing the model after tuning
grid_predictions1 = grid1.predict(X_test)# predicting the glass types for test data after tuning


# In[63]:


# Hyper peramter tuning in K nearest neighbours

from sklearn.model_selection import GridSearchCV
params =  {
    'n_neighbors': (1,10, 1),
    'leaf_size': (20,40,1),
    'p': (1,2),
    'weights': ('uniform', 'distance'),
    'metric': ('minkowski', 'chebyshev')}
grid2 = GridSearchCV(KNeighborsClassifier(),params,verbose=1)
grid2.fit(X_train,y_train)#training the model after tuning
grid_predictions2 = grid2.predict(X_test)# predicting the glass types for test data after tuning


# ### Get performance metrics for all the applied classifiers

# In[54]:


from sklearn.metrics import classification_report


# In[55]:


# classification report for support vector machine

print(classification_report(y_test,predictions))


# In[56]:


# classification report for Logistic regression

print(classification_report(y_test,predictions1))


# In[57]:


# classification report for Descision Tree

print(classification_report(y_test,predictions2))


# In[58]:


#classification report for k nearest neighbors

print(classification_report(y_test,y_pred))


# In[59]:


#classification report for svm after tuning

print(classification_report(y_test,grid_predictions))


# In[60]:


#classification report for logistic regression after tuning

print(classification_report(y_test,grid_predictions1))


# In[64]:


#classification report for k nearest neigbours after tuning

print(classification_report(y_test,grid_predictions2))


# ### Visually compare the performance of all classifiers

# In[65]:


names=["SVM","K Nearest Neighbours","Descision Trees","Logistic Regression"]
classifiers=[grid,grid2,DecisionTreeClassifier(),grid1]

# creating List and storing the accuracy scores of the 4 different classifiiers in this list
accuracyScores= []
for name,clf in zip(names,classifiers):
    clf.fit(X_train,y_train)
    score=clf.score(X_test,y_test)
    accuracyScores.append(score)


#Creating a data frame with two columns,the name of the classifier and its respective  accuracy score
df2=pd.DataFrame()
df2["Name"]=names
df2["Accuracy"]=accuracyScores
#df2

#plotting barplots for the accuracy scores of the 4 different calssifiers
sns.set(style="whitegrid")
ax=sns.barplot(y="Name",x="Accuracy",data=df2)


# after implementing and tuning the different calssifiers,we can see that k nearest neighbors model has the highest accuracy and hence it is more suitable for this data set compared to other models
