import pandas as pd
dataset = pd.read_csv('Salary_Data.csv')
print(dataset.head())
print(dataset.describe())
print(dataset.shape)
import matplotlib.pyplot as plt
plt.scatter(dataset['YearsExperience'],dataset['Salary'])
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Salary vs YearsExperience')
plt.show()


X = dataset.iloc[:,:-1].values

Y = dataset.iloc[:,-1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.linear_model import LinearRegression
model =LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color ='orange')
plt.plot(X_train,model.predict(X_train),color='yellow')
plt.title('salary vs YearsExperience (training set)')
plt.xlabel('YearsExperience')
plt.ylabel('salary')
plt.show()
plt.scatter(X_test,Y_test,color ='orange')
plt.plot(X_test,model.predict(X_test),color='yellow')
plt.title('salary vs YearsExperience (training set)')
plt.xlabel('YearsExperience')
plt.ylabel('salary')
plt.show()

