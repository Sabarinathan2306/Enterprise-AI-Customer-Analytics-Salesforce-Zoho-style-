#########################################################
# STEP 8 - FINAL CUSTOMER PROFILE (FINAL FIX)
#########################################################

import pandas as pd
import joblib

print("\nSTEP 8 STARTED")

# load data
df = pd.read_csv("outputs/clustered_data.csv")

# load models
churn_model = joblib.load("models/churn_model.pkl")
rev_model = joblib.load("models/revenue_model.pkl")

#########################################################
# LOAD TRAINING STRUCTURE
#########################################################

# columns used for churn model
X_train_churn = pd.read_csv("outputs/X_train.csv")
churn_columns = X_train_churn.columns.tolist()

# columns used for revenue model
df_preprocessed = pd.read_csv("outputs/preprocessed_data.csv")

possible_targets = [
"Total_Revenue",
"Next_Quarter_Revenue_USD",
"Revenue_Amount",
"Net_Revenue_USD"
]

revenue_target = None

for c in possible_targets:
    if c in df_preprocessed.columns:
        revenue_target = c
        break

revenue_columns = df_preprocessed.drop(columns=[revenue_target], errors="ignore").columns.tolist()

print("\nChurn model columns:")
print(churn_columns)

print("\nRevenue model columns:")
print(revenue_columns)

#########################################################
# PREPARE DATA FOR CHURN PREDICTION
#########################################################

X_churn = df.copy()

X_churn = X_churn[churn_columns]

#########################################################
# PREPARE DATA FOR REVENUE PREDICTION
#########################################################

X_revenue = df.copy()

# keep only columns used during revenue training
X_revenue = X_revenue[revenue_columns]

#########################################################
# MAKE PREDICTIONS
#########################################################

df["Churn_Probability"] = churn_model.predict_proba(X_churn)[:,1]

df["Predicted_Revenue"] = rev_model.predict(X_revenue)

#########################################################
# SAVE FINAL OUTPUT
#########################################################

df.to_csv("outputs/final_customer_profiles.csv", index=False)

print("\nFINAL CUSTOMER PROFILE CREATED")

print("\nFinal columns:")
print(df.columns.tolist())

print("\nPreview:")
print(df.head())

print("\nSaved file:")
print("outputs/final_customer_profiles.csv")

print("\nSTEP 8 COMPLETED SUCCESSFULLY")