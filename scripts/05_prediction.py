import pandas as pd
import joblib

# Load the trained model
rf_model = joblib.load('models/random_forest_model.pkl')

# Load the test data
test_df = pd.read_csv("data/test_featured.csv")

# Prepare the features (same as training features)
X_test = test_df[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Title', 'AgeBin']]

# Predict the survival outcomes
predictions = rf_model.predict(X_test)

# Create a DataFrame for submission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})

# Save the predictions to a CSV file
submission.to_csv('submissions/titanic_predictions.csv', index=False)

print("Predictions saved to 'submissions/titanic_predictions.csv'")
