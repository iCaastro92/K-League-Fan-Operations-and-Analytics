import pandas as pd
import requests
import json
import time

# 1. Load the predictions dataset
df_pred = pd.read_csv('k_league_predictions_2026.csv')

# 2. Filter target audience: free_registered AND Churn Prediction == 1
target_fans = df_pred[(df_pred['membership_type'] == 'free_registered') & 
                      (df_pred['churn_prediction'] == 1)]

# 3. Webhook URL from Make
WEBHOOK_URL = "https://hook.us2.make.com/bga2i3b4g2cuf7243h3zo97qayf0hmkw"

print(f"Found {len(target_fans)} fans at risk. Starting data transmission (PoC mode - sending first 3 records)...")

# 4. Iterate and send POST requests (simulating real-time events)
# We use .head(3) to send only 3 records for our Proof of Concept
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