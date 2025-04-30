"""
For the Titanic dataset, we'll create a few additional features, such as:

Family Size: Combining SibSp (siblings/spouses aboard) and Parch (parents/children aboard) to get the total family size.

Is Alone: A binary feature indicating whether a passenger is alone (i.e., no family members aboard).

Title: Extracting titles from the Name column (e.g., Mr, Mrs, Miss, etc.), as they can sometimes correlate with socio-economic status and survival rate.

Age Bins: Grouping age into categories (e.g., child, adult, senior).


"""

import pandas as pd

# Load cleaned datasets
train_df = pd.read_csv("data/train_cleaned.csv")
test_df = pd.read_csv("data/test_cleaned.csv")

# Feature Engineering: Create new features

# 1. Family Size = SibSp + Parch
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

# 2. Is Alone (1 if FamilySize == 0, else 0)
train_df['IsAlone'] = (train_df['FamilySize'] ==0).astype(int)
test_df['IsAlone'] = (test_df['FamilySize'] == 0).astype(int)

# 3. Title (Extract from Name)
train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
test_df['Title'] = test_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Map rare titles to 'Rare' and common titles to specific labels
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master', 
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 
    'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Rare', 'Sir': 'Rare', 
    'Jonkheer': 'Rare', 'Dona': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 
    'Mme': 'Mrs', 'Don': 'Rare', 'Baroness': 'Rare'
}

train_df['Title'] = train_df['Title'].map(title_mapping)
test_df['Title'] = test_df['Title'].map(title_mapping)

bins = [0, 12, 60, 100]
labels = ['Child', 'Adult', 'Senior']
train_df['AgeBin'] = pd.cut(train_df['Age'], bins=bins, labels=labels)
test_df['AgeBin'] = pd.cut(test_df['Age'], bins=bins, labels=labels)

# Convert categorical features to numerical values
train_df['Title'] = train_df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
test_df['Title'] = test_df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})

# AgeBin: Convert to categorical (ordinal encoding)
train_df['AgeBin'] = train_df['AgeBin'].map({'Child': 0, 'Adult': 1, 'Senior': 2})
test_df['AgeBin'] = test_df['AgeBin'].map({'Child': 0, 'Adult': 1, 'Senior': 2})

# Verify the new features
print("\nTrain Data - After Feature Engineering:")
print(train_df[['FamilySize', 'IsAlone', 'Title', 'AgeBin']].head())

# Save the updated datasets
train_df.to_csv("data/train_featured.csv", index=False)
test_df.to_csv("data/test_featured.csv", index=False)