#!/usr/bin/env python
# coding: utf-8

# In[6]:




import numpy as np 
import pandas as pd
from numpy import linalg as LA    
        
data = pd.read_csv("C:\\Users\\silicon\\Desktop\\heart.csv",sep=',')
print( data.head() )

#Get the target...
y=data.target.values #values convert values onto numpy array

#Get the predictor by excluding target
x_data = data.drop(["target"], axis=1) 

#Normalize the data...
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# fit the model with data
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n",cnf_matrix)
sum=0.0;
for i in range( len(x_test)):
    if( y_pred[i]== y_test[i]):
        sum=sum+1
Accuracy= sum/len(x_test) *100
print("Model Accuracy = ",Accuracy)  


# In[ ]:




