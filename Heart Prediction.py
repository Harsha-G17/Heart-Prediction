#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[7]:


data = pd.read_csv('Downloads/Heart Prediction Quantum Dataset.csv')


# In[8]:


data.isna().sum()


# In[10]:


data.duplicated().sum()


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


data.head()


# In[15]:


data.corr()["HeartDisease"]*100


# In[16]:


X=data.drop(columns="HeartDisease")
y=data["HeartDisease"]


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[20]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[21]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)


# In[28]:


import matplotlib.pyplot as plt
import numpy as np

feature_importance = model.feature_importances_
features = X_train.columns

sorted_idx=np.argsort(feature_importance)[::-1]
print(sorted_idx)
plt.barh(np.array(features)[sorted_idx],feature_importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Heart Disease Prediction")
plt.show()


# In[29]:


y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Model Accuracy:{accuracy:2%}")

conf_matrix = confusion_matrix(y_test,y_pred)
print("confusion Matrix:\n",conf_matrix)
print("classification Report:\n",classification_report(y_test,y_pred))


# In[31]:


feature_importance = model.feature_importances_
features=X.columns
plt.figure(figsize=(10,6))
plt.barh(features,feature_importance,color="teal")
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Cancer Prediction")
plt.show()


# In[35]:


new_data = np.array([[45,1,120,200,80,0.7]])
new_data=new_data.reshape(1,-1)
prediction =model.predict(new_data)
print("Predicted Cancer status:","Cancer" if prediction[0]==1 else "No cancer")


# In[40]:


new_data = np.array([[23,0,90,120,80,0.5]])
new_data=new_data.reshape(1,-1)
prediction =model.predict(new_data)
print("Predicted Cancer status:","Cancer" if prediction[0]==1 else "No cancer")


# In[ ]:




