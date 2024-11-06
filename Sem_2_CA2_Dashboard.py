import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
import sklearn
from skforecast.ForecasterAutoreg import ForecasterAutoreg


# Function to load data from GitHub
@st.cache
def load_data1():
    url = 'https://raw.githubusercontent.com/claireobrien00/Sem-2-CA2-Dashboard/main/AAPL.csv'
    data_AAPL = pd.read_csv(url)
    return data_AAPL

# Load the data
df_AAPL = load_data1()


# Convert 'Date' column to datetime if it's not already
df_AAPL['Date'] = pd.to_datetime(df_AAPL['Date'])

fig = go.Figure()

# Add a line trace for each price type
fig.add_trace(go.Scatter(x=df_AAPL['Date'], y=df_AAPL['Open'], mode='lines', name='Open'))
fig.add_trace(go.Scatter(x=df_AAPL['Date'], y=df_AAPL['High'], mode='lines', name='High'))
fig.add_trace(go.Scatter(x=df_AAPL['Date'], y=df_AAPL['Low'], mode='lines', name='Low'))
fig.add_trace(go.Scatter(x=df_AAPL['Date'], y=df_AAPL['Close'], mode='lines', name='Close'))

# Update layout with title and labels
fig.update_layout(
    title=" AAPL Stock Prices Over Time",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Price Type"
)

# Display the figure in Streamlit
st.plotly_chart(fig)


# Load the pre-trained model (change the path to your model)
model = joblib.load('forecaster_001.joblib')

# Streamlit app layout
st.title("Streamlit App - Predict with Loaded Model")

# Input form for user data (for example, a simple regression model)
input_data = st.text_input("Enter some data for prediction:")

# Process input and make prediction
if input_data:
    try:
        # Convert the input into a suitable format (e.g., numeric)
        # Example: If the model expects numeric input, convert input to float
        input_value = np.array([float(input_data)]).reshape(1, -1)

        # Use the loaded model to make a prediction
        prediction = model.predict(input_value)

        # Show the prediction result
        st.write(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
else:
    st.write("Please enter data to make a prediction.")


# Function to load data from GitHub
# @st.cache
# def load_data2():
#     url = 'https://raw.githubusercontent.com/claireobrien00/Sem-2-CA2-Dashboard/main/AMZN.csv'
#     data_AMZN = pd.read_csv(url)
#     return data_AMZN

# # Load the data
# df_AMZN = load_data2()


# # Convert 'Date' column to datetime if it's not already
# df_AMZN['Date'] = pd.to_datetime(df_AMZN['Date'])

# fig = go.Figure()

# # Add a line trace for each price type
# fig.add_trace(go.Scatter(x=df_AMZN['Date'], y=df_AMZN['Open'], mode='lines', name='Open'))
# fig.add_trace(go.Scatter(x=df_AMZN['Date'], y=df_AMZN['High'], mode='lines', name='High'))
# fig.add_trace(go.Scatter(x=df_AMZN['Date'], y=df_AMZN['Low'], mode='lines', name='Low'))
# fig.add_trace(go.Scatter(x=df_AMZN['Date'], y=df_AMZN['Close'], mode='lines', name='Close'))

# # Update layout with title and labels
# fig.update_layout(
#     title=" AMZN Stock Prices Over Time",
#     xaxis_title="Date",
#     yaxis_title="Price",
#     legend_title="Price Type"
# )

# # Display the figure in Streamlit
# st.plotly_chart(fig)



