#########################################################
# STEP 3 - DATA PREPROCESSING
#########################################################

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


print("\nSTEP 3 STARTED")

# load merged data
df = pd.read_csv("outputs/merged_data.csv")

print("\nColumns found in dataset:")
print(df.columns.tolist())

#########################################################
# FEATURE ENGINEERING
#########################################################

print("\nCreating new features...")

# Revenue per transaction
if "Total_Revenue" in df.columns and "Transaction_Count" in df.columns:

    df["Revenue_per_transaction"] = (
        df["Total_Revenue"] /
        (df["Transaction_Count"] + 1)
    )

# Engagement ratio
if "Engagement_Score" in df.columns:

    if "Tenure" in df.columns:

        df["Engagement_ratio"] = (
            df["Engagement_Score"] /
            (df["Tenure"] + 1)
        )

# Usage ratio
if "Usage_Score" in df.columns:

    if "Tenure" in df.columns:

        df["Usage_per_month"] = (
            df["Usage_Score"] /
            (df["Tenure"] + 1)
        )

# login behaviour
if "Avg_Login" in df.columns:

    if "Tenure" in df.columns:

        df["Login_frequency"] = (
            df["Avg_Login"] /
            (df["Tenure"] + 1)
        )

print("Feature engineering completed")

#########################################################
# HANDLE MISSING VALUES
#########################################################

print("\nHandling missing values...")

numeric_cols = df.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy="median")

df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print("Missing values handled")

#########################################################
# ENCODE CATEGORICAL VARIABLES
#########################################################

print("\nEncoding categorical columns...")

cat_cols = df.select_dtypes(include="object").columns

encoder = LabelEncoder()

for col in cat_cols:

    df[col] = encoder.fit_transform(df[col].astype(str))

print("Encoding completed")

#########################################################
# REMOVE DUPLICATES
#########################################################

df = df.drop_duplicates()

#########################################################
# FINAL CHECK
#########################################################

print("\nFinal dataset info")

print(df.info())

print("\nPreview data")

print(df.head())

#########################################################
# SAVE FILE
#########################################################

df.to_csv("outputs/preprocessed_data.csv", index=False)

print("\nSTEP 3 COMPLETED SUCCESSFULLY")

print("\nSaved file:")
print("outputs/preprocessed_data.csv")

print("\nDataset shape:", df.shape)