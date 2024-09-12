#import libraries

import requests
import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller 


app_name = 'Stock Market Analytics'
st.title(app_name)
st.subheader('This application is created to predict the stock market orice of the selected company.')

st.image("https://akm-img-a-in.tosshub.com/indiatoday/images/story/202401/top-stock-picks-for-today-the-market-veteran-also-liked-bharat-dynamics-ltd-164112329-16x9.jpg?VersionId=Xx6nIyHvkXKxieGa_KqwFIv.GCZWvK_9&size=690:388")


#sidebar
st.sidebar.header("Select Input Parameters Below")


#Taking start date and end date as input from user of this application
start_date = st.sidebar.date_input('Start Date', date(2023,11,19))
end_date = st.sidebar.date_input('End Date', date(2024,6,29))


#add ticker symbols list
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)
sp500_table = tables[0]
ticker_list = sp500_table['Symbol'].tolist()
ticker = st.sidebar.selectbox('Select the company',ticker_list)
sector = sp500_table[sp500_table['Symbol'] == ticker]['GICS Sector'].values[0]
industry = sp500_table[sp500_table['Symbol'] == ticker]['GICS Sub-Industry'].values[0]
st.write("<p style='color:red; font-size: 20px; font-weight:bold'>Entered Stock Sector :- <span style='color:green'>{}</span></p>".format(sector), unsafe_allow_html=True)
st.write("<p style='color:red; font-size: 20px; font-weight:bold'>Entered Stock Sub-Industry :- <span style='color:green'>{}</span></p>".format(industry), unsafe_allow_html=True)
 
 
#fetching data from user inputs using yfinanace library
data= yf.download(ticker, start=start_date,end=end_date)

#adding date column as data instead of index column
data.insert(0,"Date", data.index,True)
data.reset_index(drop=True, inplace=True)
st.write('----------------------------------------------Data from ', start_date,' to ', end_date,'-----------------------------------------')
st.write(data)

#plotting the data
st.header('Visualizing the data')
st.subheader('Plotting the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig = px.line(data,x='Date', y=data.columns, title='Closing price of the stock',width=2500,height=500)
st.plotly_chart(fig)

#adding a dropdown menu for selcting columns from data
column = st.selectbox('Select the column for forecasting', data.columns[1:])

data = data[['Date',column]]
st.write("Selected Data")
st.write(data)

# ADf test check for stationarity
st.header('Is data stationary')
st.write(adfuller(data[column])[1] < 0.05)

decomposition = seasonal_decompose(data[column],model='additive',period=12)
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend,title='Trend',width=2500,height=500,labels={'x':'Date','y':'Price'}))

model= sm.tsa.statespace.SARIMAX(data[column],order=(2,1,2))
model=model.fit()

st.write("<p style='color:green; font-size: 50px; font-weight:bold;'>Forecasting the data</p>", unsafe_allow_html=True)
forecast_period = st.number_input("Select the number of days to forecast",1,365,7)
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period-1)
predictions = predictions.predicted_mean

predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index)
predictions.reset_index(drop=True, inplace=True)
st.write("## Predictions",predictions)
st.write("## Actual Data", data)


fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode='lines', name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000,height=400)
st.plotly_chart(fig)

# calculate the percentage of risk,profit,loss and also show them to user at the end
actual_last_price = data[column].iloc[-1]
predicted_last_price = predictions["predicted_mean"].iloc[-1]

if actual_last_price > predicted_last_price:
    risk = ((actual_last_price - predicted_last_price) / actual_last_price) * 100
    st.header("## Risky Stock. Don't Buy This One")
    st.write(f"<p style='color:red; font-size: 20px; font-weight:bold;'>Percentage of Risk in buying that stock is : {risk:.2f}%</p>", unsafe_allow_html=True)
elif actual_last_price < predicted_last_price:
    profit = ((predicted_last_price - actual_last_price) / actual_last_price) * 100
    st.header("## Profitable Stock. You Can Buy This One")
    st.write(f"<p style='color:green; font-size: 20px; font-weight:bold;'>Percentage of Profit in buying that stock is : {profit:.2f}%</p>", unsafe_allow_html=True)
else:
    st.write("No change in price")


# Suggest top 5 similar stocks based on sector
st.header('Top 5 Similar Stocks')

# Fetch sector information of the selected ticker
sector = sp500_table[sp500_table['Symbol'] == ticker]['GICS Sector'].values[0]
industry = sp500_table[sp500_table['Symbol'] == ticker]['GICS Sub-Industry'].values[0]


# Filter stocks with the same sector and industry
similar_stocks = sp500_table[(sp500_table['GICS Sector'] == sector) & (sp500_table['Symbol'] != ticker)].head(5)
similar_stocks = similar_stocks[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

# Display the top 5 similar stocks
st.write(similar_stocks)
