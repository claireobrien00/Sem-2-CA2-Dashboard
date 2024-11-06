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
import skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg

import sys
print(sys.executable)
st.write(f"Python executable being used: {sys.executable}")


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


# # Load the pre-trained time series forecasting model
# model = joblib.load('forecaster_001.joblib')

# # Streamlit app layout
# st.title("Stock Price Prediction App")

# # Upload stock data CSV file
# uploaded_file = st.file_uploader("AAPL.csv", type="csv")

# # Process the uploaded file
# if uploaded_file is not None:
#     # Load the CSV file into a DataFrame
#     data = pd.read_csv(uploaded_file)
    
#     # Ensure the data has a 'Close' column
#     if 'Close' not in data.columns:
#         st.error("The uploaded file must contain a 'Close' column.")
#     else:
#         # Display the uploaded data
#         st.write("### Uploaded Stock Data")
#         st.write(data.tail())  # Show the last few rows of data
        
#         # Display a line chart of the close prices
#         st.write("### Close Price Over Time")
#         fig = px.line(data, x=data.index, y='Close', title="Stock Close Price History")
#         st.plotly_chart(fig)

#         # Prepare input data for prediction
#         # Assume the model requires the last 5 'Close' prices
#         last_values = data['Close'].values[-5:]  # Adjust this number based on the model's requirements

#         # Check if there are enough values for prediction
#         if len(last_values) < 5:
#             st.error("Not enough data to make a prediction. Please upload a file with at least 5 'Close' price values.")
#         else:
#             # Reshape the data to fit the model input
#             input_values = last_values.reshape(1, -1)
            
#             try:
#                 # Make a prediction for the next 'Close' price
#                 prediction = model.predict(input_values)
                
#                 # Show the prediction result
#                 st.write(f"### Predicted Next Close Price: {prediction[0]:.2f}")
            
#             except Exception as e:
#                 st.error(f"Error in prediction: {str(e)}")
# else:
#     st.write("Please upload a CSV file with stock data to make a prediction.")
# # Function to load data from GitHub
# # @st.cache
# # def load_data2():
# #     url = 'https://raw.githubusercontent.com/claireobrien00/Sem-2-CA2-Dashboard/main/AMZN.csv'
# #     data_AMZN = pd.read_csv(url)
# #     return data_AMZN

# # # Load the data
# # df_AMZN = load_data2()


# # # Convert 'Date' column to datetime if it's not already
# # df_AMZN['Date'] = pd.to_datetime(df_AMZN['Date'])

# # fig = go.Figure()

# # # Add a line trace for each price type
# # fig.add_trace(go.Scatter(x=df_AMZN['Date'], y=df_AMZN['Open'], mode='lines', name='Open'))
# # fig.add_trace(go.Scatter(x=df_AMZN['Date'], y=df_AMZN['High'], mode='lines', name='High'))
# # fig.add_trace(go.Scatter(x=df_AMZN['Date'], y=df_AMZN['Low'], mode='lines', name='Low'))
# # fig.add_trace(go.Scatter(x=df_AMZN['Date'], y=df_AMZN['Close'], mode='lines', name='Close'))

# # # Update layout with title and labels
# # fig.update_layout(
# #     title=" AMZN Stock Prices Over Time",
# #     xaxis_title="Date",
# #     yaxis_title="Price",
# #     legend_title="Price Type"
# # )

# # # Display the figure in Streamlit
# # st.plotly_chart(fig)


