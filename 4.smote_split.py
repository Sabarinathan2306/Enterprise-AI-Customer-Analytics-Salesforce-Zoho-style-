import pandas as pd

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

df=pd.read_csv("outputs/preprocessed_data.csv")

X=df.drop("Churn",axis=1)

y=df["Churn"]

smote=SMOTE()

X_res,y_res=smote.fit_resample(X,y)

X_train,X_test,y_train,y_test=train_test_split(

X_res,
y_res,
test_size=0.2,
random_state=42

)

X_train.to_csv("outputs/X_train.csv",index=False)
X_test.to_csv("outputs/X_test.csv",index=False)

y_train.to_csv("outputs/y_train.csv",index=False)
y_test.to_csv("outputs/y_test.csv",index=False)

print("SMOTE COMPLETED")