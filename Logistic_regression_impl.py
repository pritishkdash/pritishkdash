#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from numpy import linalg as LA

#Sigmoid Function...
def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))
    return y_head

def predict(beta, x_test):
    # x_test is an input for forward propagation
    z = sigmoid ( np.dot(beta.T,x_test) )
    #y_head = sigmoid(z)
    y_prediction = np.zeros((1,x_test.shape[1]))
    
    
    for i in range(z.shape[1]):
        if z[0,i] < 0.5 :
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction

#Model used to estimate beta...
def logistic_regression(x_train,y_train,theta,lernRate):
    ##Initializing of beta
    beta_old= np.full((len(x_train[0]), 1), 0.01) 
    length_x_train= len(x_train)
    #Default intialization of probabailty...
    p=np.zeros(len(x_train), dtype=float )
    
    while(True):
        for i in range(length_x_train):
            z= np.dot(beta_old.T,x_train[i])        
            p[i] = 1.0/(1.0 + np.exp(-z))     
    
        temp = lernRate * ( np.dot ( x_train.T ,(y_train-p)))
        beta_new = temp + beta_old.T
        beta_new=beta_new.T
    
        del_beta = LA.norm(beta_new - beta_old)
        beta_old= beta_new.copy()    

        if(del_beta<theta):            
            return beta_new
            break             
            
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

#Bind one col into the training set....
b = np.ones(len(x_train)) # make all values as 1

x_train=np.column_stack((b,x_train)) #bind it...
lernRate=0.01
theta=0.1

#INVOKE THE MODEL...
beta= logistic_regression(x_train,y_train,theta,lernRate)

print("Estimated beta :",beta)

#Bind one col into the training set....
a = np.ones(len(x_test))
x_test=np.column_stack((a,x_test))


#PREDICTION
y_pred=predict(beta,x_test.T)
#y_pred=predict(x_test,beta)


print("Prediction :", y_pred)
print("Actual Response",y_test)

y_pred=y_pred.T
print(y_pred.shape)
print(y_test.shape)

sum=0.0;
for i in range( len(x_test)):
    if( y_pred[i]== y_test[i]):
        sum=sum+1
Accuracy= sum/len(x_test) *100
print("Model Accuracy = ",Accuracy)        


# In[ ]:




