# import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Load Iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=1)

# Create and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)

# Predict on the test set
pred = knn.predict(xtest)

# Evaluate accuracy
accuracy = metrics.accuracy_score(ytest, pred)
print("Accuracy:", accuracy)

# Print both correct and wrong predictions
print("\nCorrect Predictions:")
for i in range(len(pred)):
    if pred[i] == ytest[i]:
        print(f"Actual: {iris.target_names[ytest[i]]}".ljust(20), f", Predicted: {iris.target_names[pred[i]]}".ljust(20))

print("\nWrong Predictions:")
for i in range(len(pred)):
    if pred[i] != ytest[i]:
        print(f"Actual: {iris.target_names[ytest[i]]}".ljust(20), f", Predicted: {iris.target_names[pred[i]]}".ljust(20))