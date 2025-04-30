import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# Load the feature-engineered datasets
train_df = pd.read_csv("data/train_featured.csv")
test_df = pd.read_csv("data/test_featured.csv")

# Prepare features (X) and target (y)
X = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Title', 'AgeBin']]
y = train_df['Survived']

# Split the training data into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the validation set
y_pred = rf_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Save the trained model to a file
joblib.dump(rf_model, 'models/random_forest_model.pkl')