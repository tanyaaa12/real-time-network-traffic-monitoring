#FINAL CODE FOR REAL TIME TRAFFIC ANALYSIS

#Real-Time Ideal Logic:
#→ Raw input → Preprocessing → PCA → Predict → Alert → Auto-Log

import pandas as pd
import joblib
import time
import os
# Load Models
scaler = joblib.load('models/scaler.pkl')
pca = joblib.load(open('models/pca.pkl', 'rb'))
model = joblib.load(open('models/cat_model.pkl', 'rb'))


def real_time_predict(raw_data):
    # Ensure correct features
    if raw_data.shape[1] > scaler.n_features_in_:
        raw_data = raw_data.iloc[:, :scaler.n_features_in_]

    # Preprocess
    processed = scaler.transform(raw_data.values)

    # PCA Transform
    pca_data = pca.transform(processed)

    # Predict
    prediction = model.predict(pca_data)

    for i, pred in enumerate(prediction):
        if pred == 1:
            print(f"Row {i+1} → 🚨 ALERT: Attack Detected!")
            with open("alerts_log.txt", "a") as f:
                f.write(f"ATTACK at {time.ctime()} \n\n")
        else:
            print(f"Row {i+1} → ✅ Safe Traffic.")

    time.sleep(5)

    raw_data['Prediction'] = prediction
    raw_data.to_csv('realtime_logs.csv', mode='a', index=False)
    
     
    return prediction


df = pd.read_csv('clean_cicids.csv')  # RAW UNTOUCHED DATA
sample = df.sample(25)
#print(sample)


real_time_predict(sample)