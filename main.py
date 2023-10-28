from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = datasets.load_iris()


# Convert the dataset to a Pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]
print(iris_df.head())

X = iris.data  # Features
y = iris.target  # Target variable (species)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


rf_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, rf_pred)
print("Accuracy:", accuracy)
rf_report = classification_report(y_test, rf_pred)

new_data = [[5.1, 3.5, 1.4, 0.2]]  # Example measurements
predicted_species = classifier.predict(new_data)
print("Predicted species:", iris.target_names[predicted_species][0])
print("Random Forest Classification Report:\n", rf_report)


