import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

app_name = 'Stock Price Prediction App'
st.title(app_name)
st.subheader('Created to forcast stock market price of the selected company')
st.image("https://media.licdn.com/dms/image/C5612AQHFS24TyOdTyA/article-cover_image-shrink_720_1280/0/1631788301339?e=1724284800&v=beta&t=Nx2SCdZYovxJd3RAH6mMKRZK8QWWhxBfm12vVkCdFkM")

st.sidebar.header('Select the parameters from below')

start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))

ticker_list = ["AAPL","MSFT","GOOG","GOOGLE","FB","TSLA","NVDA","ADBE","PYPL","INTC","CMCSA"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

data = yf.download(ticker, start=start_date, end=end_date)
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from', start_date, 'to', end_date)
st.write(data)

st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig = px.line(data, x='Date', y=data.columns, title='Closing price of the stock', width=1000, height=600)
st.plotly_chart(fig)

column = st.selectbox('Select the column to be used for forcasting', data.columns[1:])

data = data[['Date', column]]
st.write("Selected Data")
st.write(data)

st.header('Is data Stationary?')
st.write(adfuller(data[column])[1] < 0.05)

st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())

st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1200 , height=400, labels={'x':'Date', 'y': 'Price'}).update_traces(line_color = 'Blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1200 , height=400, labels={'x':'Date', 'y': 'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals', width=1200 , height=400, labels={'x':'Date', 'y': 'Price'}).update_traces(line_dash = 'dot'))

p = st.number_input("Enter p value", value=1)
d = st.number_input("Enter d value", value=1)
q = st.number_input("Enter q value", value=2)
seasonal_p = st.number_input("Enter seasonal value", value = 12)

model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,12))
model = model.fit(disp=-1)

st.header('Model Summary')
st.write(model.summary())
st.write("---")
st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data</p>", unsafe_allow_html=True)
forecast_period = st.number_input("##Enter forecast period in days", value = 10)

predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period-1)
predictions = predictions.predicted_mean

predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index)
predictions.reset_index(drop=True, inplace=True)
st.write("##Predictions", predictions)
st.write("##Actual Data", data)
st.write("---")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
st.plotly_chart(fig)

show_plots = False
if st.button('Show Separate Plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"], y=data[column], title='Actual', width=1200, height=400, labels={'x':'Date', 'y': 'Price'}).update_traces(line_color = 'Blue'))
        st.write(px.line(x=predictions["Date"], y=predictions["predicted_mean"], title='Predicted', width=1200, height=400, labels={'x':'Date', 'y': 'Price'}).update_traces(line_color='green'))
        show_plots = True
    else:
        show_plots = False
        
hide_plots = False
if st.button("Hide Separate Plots"):
    if not hide_plots:
        hide_plots = True
    else:
        hide_plots = False

st.write("---")
    