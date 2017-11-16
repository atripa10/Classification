from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import  matplotlib.pyplot as plt
# Load Data
iris=load_iris()
X= iris.data
y= iris.target

# logistic Regression
logreg = LogisticRegression()

#fit the model
logreg.fit(X,y)

#predict the respose values for observation in X
y_pred = logreg.predict(X)
print(y_pred)
print(metrics.accuracy_score(y,y_pred))


# KNN (K=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
y_pred=knn.predict(X)
print(metrics.accuracy_score(y,y_pred))

# KNN (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
y_pred=knn.predict(X)
print(metrics.accuracy_score(y,y_pred))

# spliting X and y into training and testing sets

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_train.shape)

logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

#KNN=5
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

#KNN=1
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

#KNN 1-25
scores=[]
for i in range (1,26):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)

#plotting maps

k_range=range(1,26)
plt.plot(k_range,scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()