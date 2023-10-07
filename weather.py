import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# set the dataset 
dataset = pd.read_csv('weather.data')

# now see the dataset 
dataset

# finding the missing values 
dataset.isnull().sum()

# find total number of missing values from the dataset
dataset.isnull().sum().sum()

 # how to handle the missing data, replacing with mean
dataset['Sunshine'] = dataset['Sunshine'].fillna(dataset['Sunshine'].mean())
dataset['WindSpeed9am'] = dataset['WindSpeed9am'].fillna(dataset['WindSpeed9am'].mean())

# again checking the missing values 
dataset.isnull().sum()

# extracting dependent and independent variables 
x = dataset.iloc[:,0:-4].values
y = dataset.iloc[:,21].values

# as the 16th column comes under classification so let's encode it 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_encoder_x2 = LabelEncoder()
x[: ,16] = label_encoder_x2.fit_transform(x[:,16])

# let's see the values of y 
y

# let's encode the y 
from sklearn.preprocessing import LabelEncoder
sc_y = LabelEncoder()
y = sc_y.fit_transform(y)

# encoded y 
y



x[0]

# converting the x to float datatype 
x = np.array(x , dtype = float)

# applying backward elimination
import statsmodels.api as sm

x = np.append(arr = np.ones((366, 1)).astype(int), values = x , axis = 1)

x_opt = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor_OLS.summary()

# remove the x5 and x6 column 
x_opt = x[: , [0,1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor_OLS.summary()

# # remove the x4 column 
# x_opt = x[: , [0,1,2,3,7,8,9,10,11,12,13,14,15,16,17,18]]
# regressor_OLS = sm.OLS(endog = y , exog = x_opt).fit()
# regressor_OLS.summary()

x = x_opt

print(x[0])

BE_x = x
BE_y = y

print(BE_x[0])

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
BE_x = sc_x.fit_transform(BE_x)

BE_x
BE_x[0]

# let's implement logistic regression 
# splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# step 2: fitting logistic regression to the training set 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

pickle.dump(classifier, open('weather.pkl' , 'wb'))  




  