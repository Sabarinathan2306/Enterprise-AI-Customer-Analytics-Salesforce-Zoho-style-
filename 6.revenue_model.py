#########################################################
# STEP 6 - REVENUE PREDICTION MODEL
#########################################################

import pandas as pd
import joblib
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("\nSTEP 6 STARTED")

# create models folder
os.makedirs("models", exist_ok=True)

# load dataset
df = pd.read_csv("outputs/preprocessed_data.csv")

print("\nColumns available:")
print(df.columns.tolist())

#########################################################
# AUTO DETECT REVENUE COLUMN
#########################################################

possible_targets = [
"Total_Revenue",
"Revenue_Amount",
"Net_Revenue_USD",
"Revenue",
"Amount"
]

target = None

for col in possible_targets:
    if col in df.columns:
        target = col
        break

if target is None:
    raise ValueError("Revenue column not found")

print("\nUsing revenue column:", target)

#########################################################
# PREPARE DATA
#########################################################

X = df.drop(columns=[target], errors="ignore")

y = df[target]

#########################################################
# TRAIN MODEL
#########################################################

model = GradientBoostingRegressor()

model.fit(X, y)

#########################################################
# MODEL PERFORMANCE
#########################################################

pred = model.predict(X)

print("\nMODEL PERFORMANCE")

print("MAE :", mean_absolute_error(y, pred))

print("R2  :", r2_score(y, pred))

#########################################################
# SAVE MODEL
#########################################################

joblib.dump(model, "models/revenue_model.pkl")

print("\nModel saved here:")
print("models/revenue_model.pkl")

print("\nSTEP 6 COMPLETED SUCCESSFULLY")