#Implement random forest classifier using python programming language
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier with OOB Score, Random Patches and Subspace
rf_classifier = RandomForestClassifier(oob_score=True, random_state=42, bootstrap=True, max_samples=0.8, max_features=0.8)
rf_classifier.fit(X_train, y_train)

# Print OOB score
print(f"OOB Score for Random Forest: {rf_classifier.oob_score_:.4f}")

# Make predictions with Random Forest
y_pred_rf = rf_classifier.predict(X_test)

# Train the ExtraTreesClassifier with Random Patches and Subspace
et_classifier = ExtraTreesClassifier(random_state=42, bootstrap=True, max_samples=0.8, max_features=0.8)
et_classifier.fit(X_train, y_train)

# Make predictions with ExtraTreesClassifier
y_pred_et = et_classifier.predict(X_test)

# Evaluate the Random Forest model
print("Random Forest - Confusion Matrix:")
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print(conf_matrix_rf)

print("\nRandom Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Evaluate the ExtraTreesClassifier model
print("ExtraTreesClassifier - Confusion Matrix:")
conf_matrix_et = confusion_matrix(y_test, y_pred_et)
print(conf_matrix_et)

print("\nExtraTreesClassifier - Classification Report:")
print(classification_report(y_test, y_pred_et))
