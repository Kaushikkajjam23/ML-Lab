import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Load the Iris dataset from Plotly Express for visualization (optional)
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')
fig.show()  # Visualize the Iris dataset in 3D (optional)

# Load the Iris dataset from a CSV file
dataset = pd.read_csv('Iris.csv')

# Separate features (X) and target variable (y)
X = dataset.iloc[:, 1:-1].values  # Select columns 1 to n-2 (excluding the last column)
y = dataset.iloc[:, -1].values   # Select the last column

# Encode categorical target variable (species)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Print the number of samples in the dataset
print(len(y))  # Output: 150

# Import libraries for train-test split and repeated k-fold cross-validation
from sklearn.model_selection import train_test_split, RepeatedKFold

# Split data into training and testing sets (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Use RepeatedKFold cross-validation for more robust evaluation
kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

# Perform repeated k-fold cross-validation
for train_index, test_index in kf.split(X):
    print("Train:", train_index, "Validation:", test_index)

    # Extract training and testing sets for each fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize features for better performance with some distance metrics
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Print the number of samples in the testing set for each fold
    print(len(X_test))  # Output: 30 (example for one fold)

# Create a K-Nearest Neighbors classifier with 12 neighbors and Minkowski distance (p=2)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=12, metric='minkowski', p=2)

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make a prediction on a new sample
new_sample = [[6.2, 2.2, 4.5, 1.5]]  # Example data point
predicted_class = classifier.predict(sc.transform(new_sample))
print(predicted_class)  # Output: [1] (assuming the encoded class label for Iris-Setosa)

# Evaluate the classifier's performance on the testing set (for each fold)
from sklearn.metrics import confusion_matrix, accuracy_score

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision, recall, and F1-score (macro averaging)
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
