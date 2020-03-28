#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('E:\\itsstudytym\\Python Project\\ML Notebook Sessions\\Analysis of Diabetes Prediction\\diabetes.csv')
data.head()


# In[3]:


data.shape


# In[4]:


data.isnull().sum()


# In[5]:


sns.countplot(x='Outcome',data=data)
plt.xlabel('0 means no diabetes and 1 means have diabetes')


# In[6]:


plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn')


# #### None Multicollinearity Exist

# In[7]:


print("Total number of rows : {0}".format(len(data)))
print("Number of rows missing Glucose Concentration: {0}".format(len(data.loc[data['Glucose'] == 0])))
print("Number of rows missing Blood Pressure: {0}".format(len(data.loc[data['BloodPressure'] == 0])))
print("Number of rows missing Insulin: {0}".format(len(data.loc[data['Insulin'] == 0])))
print("Number of rows missing BMI: {0}".format(len(data.loc[data['BMI'] == 0])))
print("Number of rows missing Diab_pred: {0}".format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print("Number of rows missing Age: {0}".format(len(data.loc[data['Age'] == 0])))
print("Number of rows missing Skin: {0}".format(len(data.loc[data['SkinThickness'] == 0])))


# In[8]:


data.describe()


# In[9]:


type(data)


# In[10]:


x = data.drop('Outcome',axis=1)
y = data['Outcome']


# In[11]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=0,strategy='mean')
imp_x = pd.DataFrame(imp.fit_transform(x))
imp_x.columns = x.columns
imp_index = x.index
imp_x.head()


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(imp_x,y,test_size=0.25,random_state=10)
x_train.shape


# In[13]:


x_test.shape


# ### Decision Tree Classifier

# In[14]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(x_train,y_train)


# In[15]:


y_pred = dtree.predict(x_test)
y_pred


# In[16]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
dtree_acc = accuracy_score(y_test,y_pred)
dtree_acc


# In[17]:


dtree_cm = confusion_matrix(y_test,y_pred)
dtree_cm


# In[18]:


dtree_cls = classification_report(y_test,y_pred)
print(dtree_cls)


# ### K-Nearest Neighbors Classifier

# In[19]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)


# In[20]:


yp = knn.predict(x_test)
yp


# In[21]:


knn_acc = accuracy_score(y_test,yp)
knn_acc


# In[22]:


knn_cm = confusion_matrix(y_test,yp)
knn_cm


# In[23]:


knn_cls = classification_report(y_test,yp)
print(knn_cls)


# #### Choosing K Value

# In[24]:


error_rate = []
for i in range(1,51):
    knnc = KNeighborsClassifier(n_neighbors=i)
    knnc.fit(x_train,y_train)
    pred_i = knnc.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[25]:


sns.set_style('whitegrid')
plt.figure(figsize=(12,7))
plt.plot(range(1,51),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate VS K-Value')
plt.xlabel('K-Value')
plt.ylabel('Error Rate')


# In[26]:


knnn = KNeighborsClassifier(n_neighbors=16)
knnn.fit(x_train,y_train)


# In[27]:


knny_pred = knnn.predict(x_test)
knny_pred


# In[28]:


knnn_acc = accuracy_score(y_test,knny_pred)
knnn_acc


# In[29]:


knnn_cm = confusion_matrix(y_test,knny_pred)
knnn_cm


# In[30]:


knnn_cls = classification_report(y_test,knny_pred)
print(knnn_cls)


# ### Random Forest Classifier

# In[31]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10)
rf.fit(x_train,y_train)


# In[32]:


y_predicted = rf.predict(x_test)
y_predicted


# In[33]:


rf_acc = accuracy_score(y_test,y_predicted)
rf_acc


# In[34]:


rf_cm = confusion_matrix(y_test,y_predicted)
rf_cm


# In[35]:


rf_cls = classification_report(y_test,y_predicted)
print(rf_cls)


# ### Obersvation based on Various Machine Learning Algorithms

# Decision Tree Classifier Accuracy : 0.6875 out of 1
# 

# K-Nearest Neighbors Accuracy : 0.7135 out of 1
# 

# Random Forest Classifier Accuracy : 0.75 out of 1
# 
# 

# Accuracy is not so good because of Dataset is slightly imbalaned and contains many 0 values in some rows so imputation is done using mean strategy
