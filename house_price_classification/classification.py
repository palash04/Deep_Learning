# coding: utf-8

# In[1]:

print 'Hello'


# In[2]:

import pandas as pd     # work with data as tables
import numpy as np      # use number matrices
import matplotlib.pyplot as plt
import tensorflow as tf


# In[4]:

# Load data
dataframe = pd.read_csv('data.csv') 
# remove the features we do not care about
dataframe = dataframe.drop(['index','price','sq_price'],axis=1)
# use first 10 rows
dataframe = dataframe[0:10]
dataframe


# In[7]:

# step 2 - add labels
# 1 is good buy and 0 is bad buy
dataframe.loc[:, ('y1')] = [1,1,1,0,0,1,0,1,1,1]
# y2 is a negation of y1
dataframe.loc[:, ('y2')] = dataframe['y1'] == 0 
# Turn True, false values to 1s and 0s
dataframe.loc[:, ('y2')] = dataframe['y2'].astype(int)
dataframe


# In[13]:

# step 3 - prepare for tensorflow(tensors)
# tensors are the generic version of vectors and matrices
# convert features to input tensor
inputX = dataframe.loc[:, ['area','bathrooms']].as_matrix()
# convert labels into input tensor
inputY = dataframe.loc[:, ['y1','y2']].as_matrix()


# In[14]:

inputX


# In[15]:

inputY


# In[25]:

# step 4- write our hyperparameters
learning_rate = 0.000001
training_epochs = 2000
display_step = 50
n_samples = inputY.size


# In[26]:

# step 5- create our neural network
# for feature input tensors
x = tf.placeholder(tf.float32,[None,2])
# create weights
# 2x2 float matrix
W = tf.Variable(tf.zeros([2,2]))
#add biases
b = tf.Variable(tf.zeros([2]))
# multiply weights by inputs 
# weights are how we govern how data flow in NN
y_values = tf.add(tf.matmul(x,W),b)
# apply softmax to value we just created
# softmax is our activation func
y = tf.nn.softmax(y_values)
# feed in a matrix of labels
y_ = tf.placeholder(tf.float32,[None,2])


# In[27]:

# perform training
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[28]:

# Initialize variabls and tensorflow session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# In[29]:

# training loop
for i in range(training_epochs):  
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY}) # Take a gradient descent step using our inputs and labels

    # That's all! The rest of the cell just outputs debug messages. 
    # Display logs per epoch step
    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
        print "Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc) #, \"W=", sess.run(W), "b=", sess.run(b)

print "Optimization Finished!"
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'


# In[30]:

sess.run(y, feed_dict={x: inputX })


# In[31]:

sess.run(tf.nn.softmax([1., 2.]))


# In[ ]:



