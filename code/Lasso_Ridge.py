import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
df=pd.read_csv('Data/Real-Data/Real_Combine.csv')
df=df.dropna()
X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#linear regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,X,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
#ridge regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)
print("Coefficient of determination R^2 for ridge <-- on train set: {}".format(ridge_regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 for ridge <-- on test set: {}".format(ridge_regressor.score(X_test, y_test)))
#model evaluation
ridge_prediction=ridge_regressor.predict(X_test)
# sns.distplot(y_test-ridge_prediction)
print("ridge regression best parameters")  
print(ridge_regressor.best_params_)
print("ridge regression best score")  
print(ridge_regressor.best_score_)
#lasso regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X,y)
print("Coefficient of determination R^2 <-- on train set: {}".format(lasso_regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on test set: {}".format(lasso_regressor.score(X_test, y_test)))
#model evaluation
prediction_1=lasso_regressor.predict(X_test)
prediction_2=ridge_regressor.predict(X_test)
# sns.distplot(y_test-prediction)
data1 = y_test-prediction_1
data2 = y_test-prediction_2

sns.distplot(data1, hist=True, kde=True, label='Data 1', color='red')
sns.distplot(data2, hist=True, kde=True, label='Data 2', color='blue')
plt.legend()
plt.show()
print("lasso regression best parameters")  
print(lasso_regressor.best_params_)
print("lasso regression best score")  
print(lasso_regressor.best_score_)
#regression evaluation metrics
from sklearn import metrics
print('Lasso regression MAE:', metrics.mean_absolute_error(y_test, prediction_1))
print('Lasso regression MSE:', metrics.mean_squared_error(y_test, prediction_1))
print('Lasso regression RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction_1)))

print('Ridge Regression MAE:', metrics.mean_absolute_error(y_test, prediction_2))
print('Ridge Regression MSE:', metrics.mean_squared_error(y_test, prediction_2))
print('Ridge Regression RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction_2)))
import pickle 
# open a file, where you ant to store the data
file = open('lasso_regression_model.pkl', 'wb')
# dump information to that file
pickle.dump(lasso_regressor, file)
