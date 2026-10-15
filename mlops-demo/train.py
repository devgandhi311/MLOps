from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# 1. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Accuracy:", accuracy)

# 5. Save the trained model
joblib.dump(model, "model.joblib")
print("Model saved successfully!")