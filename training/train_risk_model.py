# training/train_risk_model.py
# Simple XGBoost risk model training stub.
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

def main():
    # Placeholder synthetic features
    df = pd.DataFrame({
        'embedding_mean':[0.1,0.2,0.3,0.4],
        'clause_count':[3,5,2,6],
        'num_negations':[0,1,0,2],
        'risk':[10,70,5,80]
    })
    X = df[['embedding_mean','clause_count','num_negations']]
    y = df['risk']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = xgb.XGBRegressor(n_estimators=10).fit(X_train,y_train)
    joblib.dump(model, 'models/risk_model.joblib')

if __name__ == '__main__':
    main()
