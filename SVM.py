# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only the first two features (sepal length and sepal width) for visualization
y = iris.target  # Labels: 0 (setosa), 1 (versicolor), 2 (virginica)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Create a mesh grid for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Range for the first feature
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Range for the second feature
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))  # Create a dense grid

# Predict on the grid to visualize the decision boundary
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])  # Flatten the grid, predict for all points
Z = Z.reshape(xx.shape)  # Reshape the predictions back to the grid shape

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

# Plot the training points
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("SVM Classification with RBF Kernel")

# Add a legend for the classes
handles, labels = scatter.legend_elements()
plt.legend(handles, iris.target_names, title="Classes")

# Show the plot
plt.show()
