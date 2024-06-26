import streamlit as st
import pandas as pd
import pickle
import os

# Load the pickled Random Forest model
file_path = os.path.join(os.path.dirname(__file__), 'models', 'creditEUCard_fraudDetection.pkl')
with open(file_path, 'rb') as file:
    model = pickle.load(file)

# Define the expected number of features
EXPECTED_NUM_FEATURES = 30
EXPECTED_COLUMNS = [f'feature_{i+1}' for i in range(EXPECTED_NUM_FEATURES)]

# Function to make predictions
def predict(data):
    predictions = model.predict(data)
    return predictions

# Streamlit app
st.title('Credit Card Fraud Detection')

# Choose input method
input_method = st.radio("Choose how to input your data:", ("Upload CSV file", "Enter data manually"))

if input_method == "Upload CSV file":
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file for prediction", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", input_data)
        
        # Check if the number of columns is correct
        if input_data.shape[1] > EXPECTED_NUM_FEATURES:
            st.error(f"Error: The uploaded file has more than {EXPECTED_NUM_FEATURES} columns. Please upload a file with the correct number of columns.")
        elif input_data.shape[1] < EXPECTED_NUM_FEATURES:
            st.error(f"Error: The uploaded file has fewer than {EXPECTED_NUM_FEATURES} columns. Please upload a file with the correct number of columns.")
        else:
            # Ensure correct column order if needed
            if list(input_data.columns) != EXPECTED_COLUMNS:
                input_data.columns = EXPECTED_COLUMNS

            # Make predictions
            if st.button("Predict"):
                predictions = predict(input_data)
                prediction_labels = ["Fraudulent" if pred == 1 else "Legitimate" for pred in predictions]
                
                st.write("Predictions:")
                for i, prediction in enumerate(prediction_labels):
                    st.write(f"Transaction {i+1}: {prediction}")
                
                # Highlight fraudulent transactions
                fraudulent_transactions = [i+1 for i, pred in enumerate(predictions) if pred == 1]
                if fraudulent_transactions:
                    st.write("Fraudulent Transactions:")
                    st.write(f"Transactions {', '.join(map(str, fraudulent_transactions))} are fraudulent.")
                else:
                    st.write("No fraudulent transactions detected.")

else:
    # Manual data entry
    num_transactions = st.number_input("Enter the number of transactions:", min_value=1, step=1)
    
    if num_transactions:
        st.write(f"Please enter the values for each feature for {num_transactions} transactions (up to 12 decimal points):")
        manual_data = []
        
        for i in range(num_transactions):
            st.write(f"Transaction {i+1}")
            transaction_data = []
            for j in range(EXPECTED_NUM_FEATURES):
                value = st.number_input(f'Feature {j+1} (Transaction {i+1})', format="%.12f")
                transaction_data.append(value)
            manual_data.append(transaction_data)
        
        if st.button("Submit"):
            # Convert to DataFrame
            input_data = pd.DataFrame(manual_data, columns=EXPECTED_COLUMNS)
            
            # Make predictions
            predictions = predict(input_data)
            prediction_labels = ["Fraudulent" if pred == 1 else "Legitimate" for pred in predictions]
            
            st.write("Predictions:")
            for i, prediction in enumerate(prediction_labels):
                st.write(f"Transaction {i+1}: {prediction}")
            
            # Highlight fraudulent transactions
            fraudulent_transactions = [i+1 for i, pred in enumerate(predictions) if pred == 1]
            if fraudulent_transactions:
                st.write("Fraudulent Transactions:")
                st.write(f"Transactions {', '.join(map(str, fraudulent_transactions))} are fraudulent.")
            else:
                st.write("No fraudulent transactions detected.")
