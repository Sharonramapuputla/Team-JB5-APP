import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Streamlit app
st.title('Upload and Use Specific Pickle Models (Logistic Regression, Naive Bayes, SVC)')

# Upload the pickle files
uploaded_files = st.file_uploader("Upload model_lr.pkl, model_nb.pkl, and svc_model.pkl files", type="pkl", accept_multiple_files=True)

# Dictionary to store the models
models = {}

# Load the models
if uploaded_files:
    for file in uploaded_files:
        model_name = file.name
        if model_name in ["model_lr.pkl", "model_nb.pkl", "svc_model.pkl"]:
            models[model_name] = pickle.load(file)
    
    # Ensure all required models are uploaded
    if all(name in models for name in ["model_lr.pkl", "model_nb.pkl", "svc_model.pkl"]):
        st.success("All models loaded successfully!")
        
        # Select the model to use
        model_choices = {
            "Logistic Regression": "model_lr.pkl",
            "Naive Bayes": "model_nb.pkl",
            "Support Vector Classifier": "svc_model.pkl"
        }
        selected_model_display_name = st.selectbox('Select a model', list(model_choices.keys()))
        selected_model_name = model_choices[selected_model_display_name]
        
        if selected_model_name:
            selected_model = models[selected_model_name]
            
            # Display the model details
            st.write(f"Model '{selected_model_display_name}' loaded successfully!")
            st.write(selected_model)
            
            # Upload the test CSV file
            uploaded_csv_file = st.file_uploader("Upload the test CSV file", type="csv")
            
            if uploaded_csv_file:
                test_df = pd.read_csv(uploaded_csv_file)
                st.write("Test CSV loaded successfully!")
                
                # Display the first few rows of the test data
                st.write("First few rows of the test data:")
                st.write(test_df.head())
                
                # Assuming the test CSV does not include the target column
                # Preprocess the test data
                scaler = StandardScaler()
                test_data_scaled = scaler.fit_transform(test_df)
                
                # Predict using the selected model
                predictions = selected_model.predict(test_data_scaled)
                
                # Save predictions in a new DataFrame
                predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
                
                # Display the predictions
                st.write("Predictions:")
                st.write(predictions_df)
                
                # Provide an option to download the predictions
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )
    else:
        st.error("Please upload all three models: model_lr.pkl, model_nb.pkl, and svc_model.pkl")
