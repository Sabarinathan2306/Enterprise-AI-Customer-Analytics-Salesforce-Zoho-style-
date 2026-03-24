#########################################################
# STEP 7 - CUSTOMER SEGMENTATION (CLUSTERING)
#########################################################

import pandas as pd
import joblib
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("\nSTEP 7 STARTED")

# create folder if missing
os.makedirs("models", exist_ok=True)

# load data
df = pd.read_csv("outputs/preprocessed_data.csv")

print("\nColumns available:")
print(df.columns.tolist())

#########################################################
# AUTO SELECT CLUSTER FEATURES
#########################################################

possible_features = [

# revenue related
"Total_Revenue",
"Revenue_per_transaction",

# usage
"Usage_Score",
"Avg_Feature_Usage",
"Avg_Login",
"Login_frequency",

# engagement
"Engagement_Score",
"Engagement_ratio",

# behaviour
"Transaction_Count"

]

# select only existing columns
features = [c for c in possible_features if c in df.columns]

if len(features) < 2:
    raise ValueError("Not enough numeric columns for clustering")

print("\nUsing features for clustering:")
print(features)

#########################################################
# SCALING
#########################################################

scaler = StandardScaler()

scaled = scaler.fit_transform(df[features])

#########################################################
# KMEANS
#########################################################

kmeans = KMeans(n_clusters=4, random_state=42)

df["Cluster"] = kmeans.fit_predict(scaled)

#########################################################
# SAVE RESULTS
#########################################################

df.to_csv("outputs/clustered_data.csv", index=False)

joblib.dump(kmeans, "models/kmeans_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")

print("\nCluster distribution:")

print(df["Cluster"].value_counts())

print("\nSTEP 7 COMPLETED SUCCESSFULLY")