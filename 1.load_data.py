import pandas as pd

DATA_PATH="data"

fact_customers=pd.read_csv("fact_customers.csv")

fact_transactions=pd.read_csv("fact_transactions.csv")

fact_usage=pd.read_csv("fact_usage_monthly.csv")

fact_engagement=pd.read_csv("fact_engagement_events.csv")

print("DATA LOADED")

print(fact_customers.shape)
print(fact_transactions.shape)
print(fact_usage.shape)
print(fact_engagement.shape)

fact_customers.to_csv("outputs/customers.csv",index=False)
fact_transactions.to_csv("outputs/transactions.csv",index=False)
fact_usage.to_csv("outputs/usage.csv",index=False)
fact_engagement.to_csv("outputs/engagement.csv",index=False)