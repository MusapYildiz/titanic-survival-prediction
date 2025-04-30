import pandas as pd

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Drop the 'Cabin' column due to too many missing values
train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)

# Handle missing values in 'Age' and 'Fare' (Median Imputation)
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Handle missing values in 'Embarked' (Mode Imputation)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Ensure 'Sex' column is treated as categorical
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# Verify changes
print("\nTrain Data - Missing values after cleaning:")
print(train_df.isnull().sum())
print("\nTest Data - Missing values after cleaning:")
print(test_df.isnull().sum())

# Save cleaned data for next steps
train_df.to_csv("data/train_cleaned.csv", index=False)
test_df.to_csv("data/test_cleaned.csv", index=False)