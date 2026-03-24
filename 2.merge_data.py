import pandas as pd

# LOAD FILES
customers = pd.read_csv("outputs/customers.csv")
txn = pd.read_csv("outputs/transactions.csv")
usage = pd.read_csv("outputs/usage.csv")
eng = pd.read_csv("outputs/engagement.csv")

print("\nDATA LOADED")

print("\nUSAGE COLUMNS:")
print(usage.columns.tolist())

print("\nTRANSACTION COLUMNS:")
print(txn.columns.tolist())

print("\nENGAGEMENT COLUMNS:")
print(eng.columns.tolist())

# -------------------------
# TRANSACTION TABLE
# -------------------------

possible_revenue_cols = [
"Revenue_Amount",
"Net_Revenue_USD",
"Revenue",
"Amount",
"Transaction_Amount"
]

revenue_col = None

for c in possible_revenue_cols:
    if c in txn.columns:
        revenue_col = c
        break

if revenue_col is None:
    raise ValueError("Revenue column not found in transaction table")

txn_agg = txn.groupby("Customer_ID").agg({

revenue_col: "sum",
"Transaction_ID": "count"

}).reset_index()

txn_agg.columns = [
"Customer_ID",
"Total_Revenue",
"Transaction_Count"
]

# -------------------------
# USAGE TABLE
# -------------------------

possible_login_cols = [
"Login_Count",
"Logins",
"Login_Frequency",
"Active_Days",
"Daily_Active_Users"
]

possible_feature_cols = [
"Feature_Usage_Count",
"Feature_Usage",
"Feature_Adoption_Pct",
"Feature_Count"
]

possible_usage_score_cols = [
"Usage_Intensity_Score",
"Usage_Score",
"Usage_Intensity",
"Activity_Score"
]

login_col = next((c for c in possible_login_cols if c in usage.columns), None)

feature_col = next((c for c in possible_feature_cols if c in usage.columns), None)

usage_score_col = next((c for c in possible_usage_score_cols if c in usage.columns), None)

print("\nDETECTED USAGE COLUMNS:")
print("Login column:", login_col)
print("Feature column:", feature_col)
print("Usage score column:", usage_score_col)

usage_dict = {}

if login_col:
    usage_dict[login_col] = "mean"

if feature_col:
    usage_dict[feature_col] = "mean"

if usage_score_col:
    usage_dict[usage_score_col] = "mean"

usage_agg = usage.groupby("Customer_ID").agg(usage_dict).reset_index()

# rename columns standard format
rename_map = {}

if login_col:
    rename_map[login_col] = "Avg_Login"

if feature_col:
    rename_map[feature_col] = "Avg_Feature_Usage"

if usage_score_col:
    rename_map[usage_score_col] = "Usage_Score"

usage_agg.rename(columns=rename_map, inplace=True)

# -------------------------
# ENGAGEMENT TABLE
# -------------------------

possible_ticket_cols = [
"Support_Tickets",
"Ticket_Count",
"Complaints"
]

possible_eng_score_cols = [
"Engagement_Score",
"Engagement",
"Score"
]

ticket_col = next((c for c in possible_ticket_cols if c in eng.columns), None)

eng_score_col = next((c for c in possible_eng_score_cols if c in eng.columns), None)

print("\nDETECTED ENGAGEMENT COLUMNS:")
print("Ticket column:", ticket_col)
print("Engagement score column:", eng_score_col)

eng_dict = {}

if ticket_col:
    eng_dict[ticket_col] = "sum"

if eng_score_col:
    eng_dict[eng_score_col] = "mean"

eng_agg = eng.groupby("Customer_ID").agg(eng_dict).reset_index()

rename_map2 = {}

if ticket_col:
    rename_map2[ticket_col] = "Support_Tickets"

if eng_score_col:
    rename_map2[eng_score_col] = "Engagement_Score"

eng_agg.rename(columns=rename_map2, inplace=True)

# -------------------------
# MERGE ALL TABLES
# -------------------------

df = customers.merge(txn_agg, on="Customer_ID", how="left")

df = df.merge(usage_agg, on="Customer_ID", how="left")

df = df.merge(eng_agg, on="Customer_ID", how="left")

print("\nMERGED DATA SHAPE:", df.shape)

print("\nFINAL COLUMNS:")
print(df.columns.tolist())

df.to_csv("outputs/merged_data.csv", index=False)

print("\nSTEP 2 SUCCESSFULLY COMPLETED")