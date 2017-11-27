import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data and split to data fields and target field
df = pd.read_csv('iris_data.csv', index_col=0)

# Extracting column names
all_columns = df.columns.values
features = all_columns[:-1]
target = all_columns[-1]

# Normalizing
for feature in features:
    mean = df[feature].mean()
    std = df[feature].std()
    df[feature] = df[feature].apply(lambda x: (x - mean) / std)

# Split independent values and target values
X_data = df[df.columns.difference(['Species'])]
y_data = df[['Species']]

# One hot encoding
y_data = pd.get_dummies(y_data)

# Split data in to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=1111)

# Defining parameters
num_epochs = 500
number_of_features = X_train.shape[1]
number_of_classes = y_train.shape[1]

# Create placeholders
X_data = tf.placeholder("float", [None, number_of_features])
y = tf.placeholder("float", [None, number_of_classes])

# Create variables
W = tf.Variable(tf.zeros([number_of_features, number_of_classes]))
b = tf.Variable(tf.zeros([number_of_classes]))

# Define model (y=wx+b wrapped in a softmax)
predictions = tf.nn.softmax(tf.matmul(X_data, W) + b)

# Define the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))

# Define an optimizer
opt = tf.train.AdamOptimizer(0.01).minimize(loss)

# Open a session
with tf.Session() as sess:
    # initialize variable
    init = tf.global_variables_initializer()
    sess.run(init)

    # Train model
    for i in range(num_epochs):
        sess.run(opt, feed_dict={X_data: X_train, y: y_train})

    # Define accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1)), "float"))

    # Test model
    accuracy_value = sess.run(accuracy, feed_dict={X_data: X_test, y: y_test})
    print('Accuracy : ' + str(accuracy_value*100) + '%')

    # Print predictions
    # print(sess.run(predictions, feed_dict={X_data: X_test}))
