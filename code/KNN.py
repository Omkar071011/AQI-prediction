import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns

df=pd.read_csv('Data/Real-Data/Real_Combine.csv')

df=df.dropna()
X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor(n_neighbors=1)
regressor.fit(X_train,y_train)
KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski', n_neighbors=1, p=2,
          weights='uniform')
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on test set: {}".format(regressor.score(X_test, y_test)))
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)
score.mean()
prediction=regressor.predict(X_test)

#Hyperparameter Tuning
accuracy_rate = []
# Will take some time
for i in range(1,40):
    knn = KNeighborsRegressor(n_neighbors=i)
    score=cross_val_score(knn,X,y,cv=10,scoring="neg_mean_squared_error")
    accuracy_rate.append(score.mean())
plt.figure(figsize=(20,6))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K mValue')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')
# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train,y_train)
predictions_1 = knn.predict(X_test)
sns.displot(y_test-predictions_1)
from sklearn import metrics
print('KNN for n_neighbors = 1 MAE:', metrics.mean_absolute_error(y_test, predictions_1))
print('KNN for n_neighbors = 1 MSE:', metrics.mean_squared_error(y_test, predictions_1))
print('KNN for n_neighbors = 1 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions_1)))
# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train,y_train)
predictions_2 = knn.predict(X_test)
sns.displot(y_test-predictions_2)
plt.show()


print('KNN for n_neighbors = 3, MAE:', metrics.mean_absolute_error(y_test, predictions_2))
print('KNN for n_neighbors = 3, MSE:', metrics.mean_squared_error(y_test, predictions_2))
print('KNN for n_neighbors = 3, RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions_2)))
