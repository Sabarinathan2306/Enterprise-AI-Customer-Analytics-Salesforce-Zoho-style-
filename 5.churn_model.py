#########################################################
# STEP 5 - CHURN MODEL
#########################################################

import pandas as pd
import joblib
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


print("\nSTEP 5 STARTED")

# create models folder if not exists
os.makedirs("models", exist_ok=True)

# load data
X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")

y_train = pd.read_csv("outputs/y_train.csv")
y_test = pd.read_csv("outputs/y_test.csv")

# flatten y values
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# train model
model = GradientBoostingClassifier()

model.fit(X_train, y_train)

# prediction
pred = model.predict(X_test)

prob = model.predict_proba(X_test)[:, 1]

# metrics
print("\nMODEL PERFORMANCE")

print("Accuracy :", accuracy_score(y_test, pred))

print("ROC AUC  :", roc_auc_score(y_test, prob))

# save model
joblib.dump(model, "models/churn_model.pkl")

print("\nModel saved here:")
print("models/churn_model.pkl")

print("\nSTEP 5 COMPLETED SUCCESSFULLY")