import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# -------------------------------
# Title
# -------------------------------
st.title("📈 Stock Price Prediction using LSTM")
st.write("Upload your dataset and predict future stock prices")

# -------------------------------
# Upload File
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Load data
    df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # REMOVE NON-NUMERIC COLUMNS (FIX)
    # -------------------------------
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        st.error("❌ No numeric columns found in dataset")
        st.stop()

    # Select column
    column = st.selectbox("Select column for prediction", numeric_df.columns)

    data = numeric_df[[column]].values

    # -------------------------------
    # Plot Original Data
    # -------------------------------
    st.subheader("📉 Original Data")
    fig1 = plt.figure()
    plt.plot(data.flatten())
    plt.title("Original Data")
    st.pyplot(fig1)

    # -------------------------------
    # Scaling
    # -------------------------------
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # -------------------------------
    # Create Dataset
    # -------------------------------
    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(time_step, len(dataset)):
            X.append(dataset[i-time_step:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # -------------------------------
    # Build Model
    # -------------------------------
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # -------------------------------
    # Train Model
    # -------------------------------
    st.write("⏳ Training model... Please wait")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # -------------------------------
    # Predictions
    # -------------------------------
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)

    y_train_actual = scaler.inverse_transform(y_train.reshape(-1,1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

    # -------------------------------
    # Train Graph
    # -------------------------------
    st.subheader("📊 Training Prediction")
    fig2 = plt.figure()
    plt.plot(y_train_actual.flatten(), label="Actual")
    plt.plot(train_pred.flatten(), label="Predicted")
    plt.legend()
    st.pyplot(fig2)

    # -------------------------------
    # Test Graph
    # -------------------------------
    st.subheader("📊 Testing Prediction")
    fig3 = plt.figure()
    plt.plot(y_test_actual.flatten(), label="Actual")
    plt.plot(test_pred.flatten(), label="Predicted")
    plt.legend()
    st.pyplot(fig3)

    # -------------------------------
    # Future Prediction (30 Days)
    # -------------------------------
    last_60 = scaled_data[-60:]
    future_input = last_60.reshape(1,60,1)

    future_predictions = []

    for i in range(30):
        pred = model.predict(future_input, verbose=0)
        future_predictions.append(pred[0][0])

        pred_reshaped = pred.reshape(1,1,1)
        future_input = np.concatenate((future_input[:,1:,:], pred_reshaped), axis=1)

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1,1)
    )

    # -------------------------------
    # Future Graph
    # -------------------------------
    st.subheader("🔮 Future 30 Days Prediction")
    fig4 = plt.figure()
    plt.plot(future_predictions.flatten(), label="Future Prediction")
    plt.legend()
    st.pyplot(fig4)

    # -------------------------------
    # Business Insights
    # -------------------------------
    st.subheader("🧠 Business Insights")

    st.write("""
    ✔ Helps investors decide buy/sell timing  
    ✔ Identifies stock trends and patterns  
    ✔ Useful for portfolio management  
    ✔ Supports data-driven decision making  
    ✔ Reduces investment risk using predictions  
    """)

# Footer
st.write("✅ Project by You | LSTM + Streamlit")