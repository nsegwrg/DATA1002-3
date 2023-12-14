#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from math import sqrt
from sklearn import metrics
from sklearn import neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv('SDG_goal3_clean.csv')
umm = pd.get_dummies(df["Region"])
df = pd.concat([df,umm],axis = 1,sort = False)
phys = df.drop(columns = ["Country","Year","Region","Maternal mortality ratio"])     
mort = df['Maternal mortality ratio']        
X_train, X_test, y_train, y_test = train_test_split(phys, mort, test_size=0.1, random_state=42)
neigh = neighbors.KNeighborsRegressor(n_neighbors=4).fit(X_train.values, y_train.values)

sample = [99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71]        
sample_pred = neigh.predict([sample])
print('Predicted maternal mortality:', int(sample_pred))

# Use the model to predict X_test
y_pred = neigh.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
print('Root mean squared error (RMSE):', int(mse))
print('R-squared score:', metrics.r2_score(y_test, y_pred))
print("mean abs error : ", int(mae))


# In[24]:


from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
max_r2 = 0
x = list(range(0,17))
if (r2_score(y_test, y_pred) > max_r2):
      best_y_pred = y_pred
      best_y_test = y_test
      max_r2 = max(max_r2, r2_score(y_test, y_pred))

plt.plot(x, best_y_pred, c = 'orange')
plt.plot(x, best_y_test, c = 'red')
plt.legend(['Predicted value', 'Actual value'], bbox_to_anchor=(1.4, 1))
plt.xlabel('Observations in the test split')
plt.ylabel('Maternal mortality ratio')
plt.title('Line plot of actual vs model predicted values of MMR\n')


# In[ ]:




