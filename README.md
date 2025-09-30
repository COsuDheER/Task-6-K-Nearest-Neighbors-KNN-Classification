# Task-6-K-Nearest-Neighbors-KNN-Classification

K-Nearest Neighbors (KNN) Classification Project: Iris Dataset

Project Objective

The main goal of this project was to understand and implement the K-Nearest Neighbors (KNN) algorithm for a fundamental classification problem. KNN is a non-parametric, lazy learning algorithm used for classification and regression.

Tools and Libraries

This project utilized standard data science and machine learning libraries in Python:

Scikit-learn (sklearn): For loading the dataset, scaling features, implementing the KNN model, and calculating performance metrics.

Pandas: Used for data handling and structure (though mainly NumPy arrays were utilized for processing).

Matplotlib & Seaborn: Used for data visualization, including plotting the Error Rate vs. K curve and visualizing the Confusion Matrix and Decision Boundaries.

Description: This dataset contains 150 samples of iris flowers, with four features measured from each sample: sepal length, sepal width, petal length, and petal width. The task is to classify the species into one of three classes: Iris setosa, Iris versicolor, or Iris virginica.

Task Breakdown and Implementation
The project was executed in five sequential tasks to cover the entire machine learning workflow:

Task 1: Data Preparation and Feature Normalization

KNN is a distance-based algorithm, making it highly sensitive to the scale of features.

The Iris data was loaded and split into training and testing sets (70% train / 30% test).

The features were normalized using StandardScaler to ensure all features contribute equally to the distance calculation. Scaling was fitted only on the training data and applied to both sets.

Task 2: Model Initialization and Training

The KNeighborsClassifier from sklearn was initialized with an arbitrary starting value of K=5.

The model was trained (.fit()) using the normalized training data.

Initial predictions were made on the normalized test data (.predict()).

Task 3: Experimentation and Finding the Optimal K

The optimal number of neighbors (K) is crucial for KNN performance.

The model was tested across a range of K values, from K=1 to K=20.

The error rate (1 - accuracy) on the test set was calculated for each K.

A plot of Error Rate vs. K Value was generated to visually identify the point of lowest error, determining the optimal K (K 
best) to be used for the final model.

Task 4: Model Evaluation

The final model was created using the determined optimal K, and its performance was rigorously evaluated on the test set.

Accuracy: The overall percentage of correct predictions was calculated.

Confusion Matrix: A matrix was generated and visualized using seaborn to show the true vs. predicted counts for each of the three species classes. This helps identify where the model is making errors.

Classification Report: A detailed report was generated, providing Precision, Recall, and F1-Score for each individual class.

Task 5: Visualization of Decision Boundaries

To understand how the KNN model segments the feature space, the decision boundaries were visualized.

Due to the 4D nature of the data, the model was retrained using only the Sepal Length and Sepal Width features.

The model was used to predict the class for a dense mesh across the 2D feature space.

The resulting plot displays the distinct regions (boundaries) classified by the KNN model, overlaid with the actual data points, providing a clear geometric interpretation of the algorithm's predictions.
