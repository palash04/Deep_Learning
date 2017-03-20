import pandas as pd 	#lets read our dataset
from sklearn import linear_model	#machine learning library
# matplotlib - let's us visualize model and data
import matplotlib.pyplot as plt 

#read data
dataFrame = pd.read_fwf('brain_body.txt')
x_values = dataFrame[['Brain']]
y_values = dataFrame[['Body']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values,y_values)

#visualize results
plt.scatter(x_values,y_values)
plt.plot(x_values,body_reg.predict(x_values))
plt.show()
 
