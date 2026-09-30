import os
import pandas as pd
import requests
import json
import time

# 1. Securely fetch the Webhook URL from the environment
WEBHOOK_URL = os.environ.get('MAKE_WEBHOOK_URL')

# 2. Safety check: Stop execution if the secret is missing
if not WEBHOOK_URL:
    raise ValueError("CRITICAL ERROR: MAKE_WEBHOOK_URL environment variable is not set. Cannot send data.")

# 3. Load the predictions dataset
# (Ensure this file is pushed to your GitHub repository later)
df_pred = pd.read_csv('k_league_predictions_2026.csv')

# 4. Filter target audience: free_registered AND Churn Prediction == 1
target_fans = df_pred[(df_pred['membership_type'] == 'free_registered') & 
                      (df_pred['churn_prediction'] == 1)]

print(f"Found {len(target_fans)} fans at risk. Starting data transmission (PoC mode - sending first 3 records)...")

# 5. Iterate and send POST requests
for index, row in target_fans.head(3).iterrows(): 
    payload = {
        "fan_id": row['fan_id'],
        "clv_2026_krw": row['clv_2026_krw'],
        "membership_type": row['membership_type'],
        "churn_prediction": row['churn_prediction']
    }
    
    # Send the HTTP POST request
    response = requests.post(
        WEBHOOK_URL, 
        data=json.dumps(payload), 
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Sent Fan ID: {row['fan_id']} | Status Code: {response.status_code}")
    time.sleep(2) # 2-second pause to simulate organic event flow