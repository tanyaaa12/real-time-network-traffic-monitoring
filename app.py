import streamlit as st
import pandas as pd
import joblib
import time

st.title("🚀 Real-Time Intrusion Detection System")

scaler = joblib.load('models/scaler.pkl')
pca = joblib.load(open('models/pca.pkl', 'rb'))
model = joblib.load(open('models/cat_model.pkl', 'rb'))

st.markdown("### Upload your Network Traffic Data (CSV)")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    st.write("### Displaying Random 30 Rows from Uploaded Data")
    
    raw_data.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    raw_data.dropna(inplace=True)

    random_30 = raw_data.sample(30) if len(raw_data) > 30 else raw_data
    st.dataframe(random_30)

    if st.button("Start Real-Time Prediction"):
        progress = st.progress(0)
        
        processed = scaler.transform(raw_data.values[:, :scaler.n_features_in_])
        pca_data = pca.transform(processed)
        prediction = model.predict(pca_data)
        
        raw_data['Prediction'] = prediction
        
        for i, pred in enumerate(random_30.index):
            label = raw_data.loc[pred, 'Prediction']
            if label == 1:
                st.error(f"Row {i+1} → 🚨 ALERT Detected!")
            else:
                st.success(f"Row {i+1} → ✅ Safe Traffic.")
            progress.progress((i+1) / len(random_30))
            time.sleep(0.5)

        raw_data.to_csv('realtime_logs.csv', index=False)
        st.markdown("Complete Logs saved as realtime_logs.csv")

        st.download_button(
            label="Download Full Logs",
            data=raw_data.to_csv().encode('utf-8'),
            file_name='realtime_logs.csv',
            mime='text/csv'
        )



        
    