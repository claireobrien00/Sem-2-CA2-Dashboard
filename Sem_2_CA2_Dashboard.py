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
# import skforecast
# from skforecast.ForecasterAutoreg import ForecasterAutoreg

# import sys
# print(sys.executable)
# st.write(f"Python executable being used: {sys.executable}")


# Function to load data from GitHub
@st.cache_data
def load_data1():
    url = 'https://raw.githubusercontent.com/claireobrien00/Sem-2-CA2-Dashboard/main/AAPL.csv'
    data_AAPL = pd.read_csv(url)
    return data_AAPL

# Load the data
df_AAPL = load_data1()


# Convert 'Date' column to datetime if it's not already
df_AAPL['Date'] = pd.to_datetime(df_AAPL['Date'])


# Streamlit select box for price type
price_type = st.selectbox(
    "Select Price Type to Display",
    options=["Open", "High", "Low", "Close"]
)

# Initialize the Plotly figure
fig = go.Figure()

# Add a line trace based on the selected price type
fig.add_trace(go.Scatter(x=df_AAPL['Date'], y=df_AAPL[price_type], mode='lines', name=price_type))

# Update layout with title and labels
fig.update_layout(
    title="AAPL Stock Prices Over Time",
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
# Function to load data from GitHub
def load_data2():
    url = 'https://raw.githubusercontent.com/claireobrien00/Sem-2-CA2-Dashboard/main/AMZN.csv'
    data_AMZN = pd.read_csv(url)
    return data_AMZN

# Load the data
df_AMZN = load_data2()

# Convert 'Date' column to datetime if it's not already
df_AMZN['Date'] = pd.to_datetime(df_AMZN['Date'])

# Extract unique months and years from the data
df_AMZN['Year'] = df_AMZN['Date'].dt.year
df_AMZN['Month'] = df_AMZN['Date'].dt.month
months = df_AMZN['Month'].unique()
years = df_AMZN['Year'].unique()

# Create a dropdown in Streamlit for year and month selection
selected_year = st.selectbox("Select Year", sorted(years, reverse=True))
selected_month = st.selectbox("Select Month", sorted(months))

# Filter data based on selected month and year
filtered_data = df_AMZN[(df_AMZN['Year'] == selected_year) & (df_AMZN['Month'] == selected_month)]

# Initialize the Plotly figure
fig = go.Figure()

# Add a line trace for each price type
fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Open'], mode='lines', name='Open'))
fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['High'], mode='lines', name='High'))
fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Low'], mode='lines', name='Low'))
fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Close'], mode='lines', name='Close'))

# Update layout with title and labels
fig.update_layout(
    title=f"AMZN Stock Prices for {selected_year}-{selected_month:02d}",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Price Type"
)

# Display the figure in Streamlit
st.plotly_chart(fig)



def load_data3():
    url = 'https://raw.githubusercontent.com/claireobrien00/Sem-2-CA2-Dashboard/main/BA.csv'
    data_BA = pd.read_csv(url)
    return data_BA



# Load the data
df_BA = load_data3()


# Convert 'Date' column to datetime if it's not already
df_BA['Date'] = pd.to_datetime(df_BA['Date'])

fig = go.Figure()

# Add a line trace for each price type
fig.add_trace(go.Scatter(x=df_BA['Date'], y=df_BA['Open'], mode='lines', name='Open'))
fig.add_trace(go.Scatter(x=df_BA['Date'], y=df_BA['High'], mode='lines', name='High'))
fig.add_trace(go.Scatter(x=df_BA['Date'], y=df_BA['Low'], mode='lines', name='Low'))
fig.add_trace(go.Scatter(x=df_BA['Date'], y=df_BA['Close'], mode='lines', name='Close'))

# Update layout with title and labels
fig.update_layout(
    title=" BA Stock Prices Over Time",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Price Type"
)

# Display the figure in Streamlit
st.plotly_chart(fig)



def load_data4():
    url = 'https://raw.githubusercontent.com/claireobrien00/Sem-2-CA2-Dashboard/main/DIS.csv'
    data_DIS = pd.read_csv(url)
    return data_BA



# Load the data
df_DIS = load_data3()


# Convert 'Date' column to datetime if it's not already
df_DIS['Date'] = pd.to_datetime(df_DIS['Date'])

fig = go.Figure()

# Add a line trace for each price type
fig.add_trace(go.Scatter(x=df_DIS['Date'], y=df_DIS['Open'], mode='lines', name='Open'))
fig.add_trace(go.Scatter(x=df_DIS['Date'], y=df_DIS['High'], mode='lines', name='High'))
fig.add_trace(go.Scatter(x=df_DIS['Date'], y=df_DIS['Low'], mode='lines', name='Low'))
fig.add_trace(go.Scatter(x=df_DIS['Date'], y=df_DIS['Close'], mode='lines', name='Close'))

# Update layout with title and labels
fig.update_layout(
    title=" DIS Stock Prices Over Time",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Price Type"
)

# Display the figure in Streamlit
st.plotly_chart(fig)





def load_data5():
    url = 'https://raw.githubusercontent.com/claireobrien00/Sem-2-CA2-Dashboard/main/TSLA.csv'
    data_TSLA = pd.read_csv(url)
    return data_TSLA



# Load the data
df_TSLA = load_data5()


# Convert 'Date' column to datetime if it's not already
df_TSLA['Date'] = pd.to_datetime(df_TSLA['Date'])

fig = go.Figure()

# Add a line trace for each price type
fig.add_trace(go.Scatter(x=df_TSLA['Date'], y=df_TSLA['Open'], mode='lines', name='Open'))
fig.add_trace(go.Scatter(x=df_TSLA['Date'], y=df_TSLA['High'], mode='lines', name='High'))
fig.add_trace(go.Scatter(x=df_TSLA['Date'], y=df_TSLA['Low'], mode='lines', name='Low'))
fig.add_trace(go.Scatter(x=df_TSLA['Date'], y=df_TSLA['Close'], mode='lines', name='Close'))

# Update layout with title and labels
fig.update_layout(
    title=" TSLA Stock Prices Over Time",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Price Type"
)

# Display the figure in Streamlit
st.plotly_chart(fig)


fig = go.Figure()

# Add a line trace for each price type
fig.add_trace(go.Scatter(x=df_TSLA['Date'], y=df_TSLA['Close'], mode='lines', name='TSLA'))
fig.add_trace(go.Scatter(x=df_DIS['Date'], y=df_DIS['Close'], mode='lines', name='DIS'))
fig.add_trace(go.Scatter(x=df_BA['Date'], y=df_BA['Close'], mode='lines', name='BA'))
fig.add_trace(go.Scatter(x=df_AMZN['Date'], y=df_AMZN['Close'], mode='lines', name='AMZN'))
fig.add_trace(go.Scatter(x=df_AAPL['Date'], y=df_AAPL['Close'], mode='lines', name='AAPL'))

# Update layout with title and labels
fig.update_layout(
    title="Five companies closing stock prices for 2020",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Price Type"
)

# Display the figure in Streamlit
st.plotly_chart(fig)

