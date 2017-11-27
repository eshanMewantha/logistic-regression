from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Load and split data to data fields and target field
iris = load_iris()
X, y = iris.data[:-1, :], iris.target[:-1]

# Split data to train and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

# Create model
logistic = LogisticRegression()

# Train model
logistic.fit(train_X, train_y)

# Test model
predictions = logistic.predict(test_X)

# Evaluate performance
print('Accuracy     : ' + str(accuracy_score(test_y, predictions) * 100) + '%')
print('Precision    : ' + str(precision_score(test_y, predictions, average='weighted') * 100) + '%')
print('Recall       : ' + str(recall_score(test_y, predictions, average='weighted') * 100) + '%')
