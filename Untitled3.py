#!/usr/bin/env python
# coding: utf-8

# # Project: Basic Image Classification

# In[8]:


#importing libraries
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


#importing data from keras.datasets
from tensorflow.keras.datasets import mnist

#data in the form of training as well as testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[10]:


#checking the shapes of dataset
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)


# In[11]:


#ploting one of the image of dataset
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(x_train[5], cmap = 'binary')
plt.show()


# In[12]:


#cehecking y_train set
print(set(y_train))


# In[13]:


#As this y_train and y_test are the numpy arrays that represents the digit in x set
#now our task is to ENCODE each value into a 10-dimensional vector
#that represents each of the digit
from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)


# In[14]:


print('y_train shape: ', y_train_encoded.shape)
print('y_test shape: ', y_test_encoded.shape)


# In[20]:


#checking how y_train and y_train_encoded are related
for i in range(5):
    print(y_train[i],y_train_encoded[i])


# In[23]:


#Preprocessing
#now we create a neural network
#unwrapping the x-(28,28) to x-(784,1)
x_train_reshaped = np.reshape(x_train,(60000,784))
x_test_reshaped = np.reshape(x_test,(10000,784))

print('x_train_reshaped shape: ', x_train_reshaped.shape)
print('x_test_reshaped shape: ', x_test_reshaped.shape)


# In[25]:


#Pixel values range from 0 to 255
print(set(x_train_reshaped[0]))


# In[30]:


#We normalize these values to fit in the model
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

print('mean: ', x_mean)
print('std: ', x_std)

epsilon = 1e-10
x_train_norm = (x_train_reshaped - x_mean)/(x_std+epsilon)
x_test_norm = (x_test_reshaped - x_mean)/(x_std+epsilon)

print(set(x_train_norm[0]))


# # Creating a model
# 

# In[31]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation = 'relu', input_shape = (784,)),
    Dense(128, activation= 'relu'),
    Dense(10, activation = 'softmax')
])


# In[32]:


model.compile(
    optimizer = 'sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()


# In[33]:


h = model.fit(
    x_train_norm,
    y_train_encoded,
    epochs = 5
)


# In[34]:


#testing on test data
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('test set accuracy: ', accuracy*100)


# In[36]:


#making predictions
preds = model.predict(x_test_norm)
print('shape of preds: ', preds.shape)


# In[40]:


#plot

plt.figure(figsize= (12,12))

start_index = 0

for i in range(36):
    plt.subplot(6,6,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    pred = np.argmax(preds[start_index + i])
    actual = np.argmax(y_test_encoded[start_index + i])
    col = 'g'
    if pred != actual:
        col = 'r'
    plt.xlabel('i={} | pred={} | true={}'.format(start_index +i,pred,actual),color = col)
    plt.imshow(x_test[start_index + i],cmap = 'binary')
plt.show()


# In[45]:


count = 0
for i in range(9999):
    pred = np.argmax(preds[i])
    actual = np.argmax(y_test_encoded[i])
    if pred!= actual:
        count += 1
print(count)
#INFERENCE
#just to check how many predictions went wrong
#and we can see the accuracy. out of 10000, only 301 are incorrect
#model has 97% accuracy.


# In[ ]:




