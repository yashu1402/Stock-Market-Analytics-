import requests
import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# App name
app_name = 'Stock Market Analytics'
st.title(app_name)
st.subheader('This application is created to predict the stock market price of the selected company.')

# Display an image
st.image("https://akm-img-a-in.tosshub.com/indiatoday/images/story/202401/top-stock-picks-for-today-the-market-veteran-also-liked-bharat-dynamics-ltd-164112329-16x9.jpg?VersionId=Xx6nIyHvkXKxieGa_KqwFIv.GCZWvK_9&size=690:388")

# Function to fetch top gainers and losers with company name and sector
def fetch_top_gainers_losers():
    # Fetch S&P 500 company tickers
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]
    
    tickers = sp500_table['Symbol'].tolist()
    sectors = sp500_table.set_index('Symbol')['GICS Sector'].to_dict()
    company_names = sp500_table.set_index('Symbol')['Security'].to_dict()
    
    # Fetch current stock prices
    data = yf.download(tickers, period='1d', group_by='ticker', auto_adjust=True)
    
    # Debug: Check if data is fetched
    print(data.head())

    # Calculate the percentage change
    changes = {}
    current_prices = {}
    last_prices = {}
    for ticker in tickers:
        if ticker in data.columns:
            if not data[ticker].empty and 'Close' in data[ticker] and 'Open' in data[ticker]:
                try:
                    change_percent = (data[ticker]['Close'][-1] - data[ticker]['Open'][-1]) / data[ticker]['Open'][-1] * 100
                    changes[ticker] = change_percent
                    current_prices[ticker] = data[ticker]['Close'][-1]
                    last_prices[ticker] = data[ticker]['Open'][-1]
                except IndexError:
                    continue  # Skip if there's an IndexError

    # Debug: Check changes dictionary
    print("Changes Dictionary:", changes)

    # Create DataFrame and filter valid changes
    change_df = pd.DataFrame(list(changes.items()), columns=['Ticker', 'Change'])
    change_df['Company'] = change_df['Ticker'].map(company_names)
    change_df['Sector'] = change_df['Ticker'].map(sectors)
    change_df['Current Price'] = change_df['Ticker'].map(current_prices)
    change_df['Last Price'] = change_df['Ticker'].map(last_prices)

    # Convert 'Change' to numeric
    change_df['Change'] = pd.to_numeric(change_df['Change'], errors='coerce')
    change_df.dropna(subset=['Change'], inplace=True)  # Drop rows where Change is NaN

    # Debug: Check change_df before filtering
    print("Change DataFrame before filtering:", change_df)

    # Add percentage change column for display
    change_df['Percent Change'] = change_df['Change'].apply(lambda x: f"{x:.2f}%")
    
    # Get top gainers and losers
    if not change_df.empty:
        top_gainers = change_df.nlargest(5, 'Change')
        top_losers = change_df.nsmallest(5, 'Change')
    else:
        top_gainers = pd.DataFrame(columns=['Company', 'Ticker', 'Change', 'Sector', 'Percent Change', 'Current Price', 'Last Price'])
        top_losers = pd.DataFrame(columns=['Company', 'Ticker', 'Change', 'Sector', 'Percent Change', 'Current Price', 'Last Price'])

    return top_gainers, top_losers

# Display live market top gainers and losers
st.header('Live Market Top Gainers and Losers')
gainers, losers = fetch_top_gainers_losers()

st.subheader('Top Gainers')
gainers_display = gainers[['Company', 'Ticker', 'Change', 'Sector', 'Percent Change', 'Current Price', 'Last Price']]
gainers_display.columns = ['Company', 'Ticker', 'Change', 'Sector', 'Percent Gain', 'Current Price', 'Last Price']
st.write(gainers_display)

st.subheader('Top Losers')
losers_display = losers[['Company', 'Ticker', 'Change', 'Sector', 'Percent Change', 'Current Price', 'Last Price']]
losers_display.columns = ['Company', 'Ticker', 'Change', 'Sector', 'Percent Loss', 'Current Price', 'Last Price']
st.write(losers_display)

# Function to fetch monthly performance of stocks
#def fetch_monthly_performance():
    # Fetching S&P 500 tickers
    #url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    #tables = pd.read_html(url)
    #sp500_table = tables[0]
    
    #tickers = sp500_table['Symbol'].tolist()
    #sectors = sp500_table.set_index('Symbol')['GICS Sector'].to_dict()
    #company_names = sp500_table.set_index('Symbol')['Security'].to_dict()

    # Define the time period (last 30 days)
    #end_date = date.today()
    #start_date = end_date - timedelta(days=30)

    # Download stock data for the last month
    #data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    # Calculate monthly percentage change
    #monthly_changes = {}
    #current_prices = {}
    #last_month_prices = {}

     #for ticker in tickers:
       # if ticker in data.columns and not data[ticker].empty and 'Close' in data[ticker]:
          #  try:
         #       change_percent = (data[ticker]['Close'][-1] - data[ticker]['Close'][0]) / data[ticker]['Close'][0] * 100
        #        monthly_changes[ticker] = change_percent
       #         current_prices[ticker] = data[ticker]['Close'][-1]
      #          last_month_prices[ticker] = data[ticker]['Close'][0]
     #       except IndexError:
    #            continue  # Skip if there's an IndexError

    # Create DataFrame for monthly performance
   # monthly_df = pd.DataFrame(list(monthly_changes.items()), columns=['Ticker', 'Change'])
   # monthly_df['Company'] = monthly_df['Ticker'].map(company_names)
   # monthly_df['Sector'] = monthly_df['Ticker'].map(sectors)
   # monthly_df['Current Price'] = monthly_df['Ticker'].map(current_prices)
  #  monthly_df['Last Month Price'] = monthly_df['Ticker'].map(last_month_prices)

    # Convert 'Change' to numeric and drop NaN values
 #   monthly_df['Change'] = pd.to_numeric(monthly_df['Change'], errors='coerce')
#    monthly_df.dropna(subset=['Change'], inplace=True)

  # Add percentage change column
 #   monthly_df['Percent Change'] = monthly_df['Change'].apply(lambda x: f"{x:.2f}%")

#    return monthly_df

# Display monthly performance
#st.header('Monthly Performance of Stocks')
#monthly_performance = fetch_monthly_performance()

# Display the top gainers and losers for the last month
#st.subheader('Top 5 Monthly Gainers')
#monthly_gainers = monthly_performance.nlargest(5, 'Change')
#st.write(monthly_gainers[['Company', 'Ticker', 'Sector', 'Percent Change', 'Current Price', 'Last Month Price']])

#st.subheader('Top 5 Monthly Losers')
#monthly_losers = monthly_performance.nsmallest(5, 'Change')
#st.write(monthly_losers[['Company', 'Ticker', 'Sector', 'Percent Change', 'Current Price', 'Last Month Price']]) 

# Sidebar header
st.sidebar.header("Select Input Parameters Below")

# Adding a button to refresh data
refresh_button = st.sidebar.button("Refresh Data")

# Taking start date and end date as input from user
start_date = st.sidebar.date_input('Start Date', date.today() - timedelta(days=365))
end_date = st.sidebar.date_input('End Date', date.today())

# Fetching S&P 500 company tickers
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)
sp500_table = tables[0]
ticker_list = sp500_table['Symbol'].tolist()
ticker = st.sidebar.selectbox('Select the company', ticker_list)

# Add space or divider to visually separate the footer section
st.sidebar.markdown("---")  # A horizontal line (divider)

# Collapsible "About Us" Section
with st.sidebar.expander("About Us"):
    st.write("""
    Welcome to Stock Market Analytics, your one-stop platform for tracking and analyzing the stock market. 
    Our mission is to provide users with a comprehensive and user-friendly interface to explore the world of finance.
    Our team of experts is dedicated to delivering accurate and timely data, along with powerful analytics tools, to help you make informed investment decisions.
    Whether you're a seasoned investor or just starting out, we're committed to helping you navigate the complex world of finance with ease.
    """)

# Collapsible "Contact Us" Section
with st.sidebar.expander("Contact Us"):
    st.write("""
    Feel free to reach out to us for any inquiries or support:
    - **Email**: support@stocktracker.com
    - **Phone**: +1 234 567 890
    """)

# Fetch sector and industry for the selected stock
sector = sp500_table[sp500_table['Symbol'] == ticker]['GICS Sector'].values[0]
industry = sp500_table[sp500_table['Symbol'] == ticker]['GICS Sub-Industry'].values[0]
st.write(f"<p style='color:red; font-size: 20px; font-weight:bold'>Entered Stock Sector :- <span style='color:green'>{sector}</span></p>", unsafe_allow_html=True)
st.write(f"<p style='color:red; font-size: 20px; font-weight:bold'>Entered Stock Sub-Industry :- <span style='color:green'>{industry}</span></p>", unsafe_allow_html=True)

# Refresh data if refresh button is pressed
if refresh_button:
    data = yf.download(ticker, start=start_date, end=end_date)
else:
    data = yf.download(ticker, start=start_date, end=end_date)

# Adding a Date column and resetting index
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write(f'----------------------------------------------Data from {start_date} to {end_date}-----------------------------------------')
st.write(data)

# Plotting the data
st.header('Visualizing the data')
st.subheader('Plotting the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig = px.line(data, x='Date', y=data.columns, title='Closing price of the stock', width=2500, height=500)
st.plotly_chart(fig)

# Adding a dropdown menu for selecting columns from data
column = st.selectbox('Select the column for forecasting', data.columns[1:])
data = data[['Date', column]]
st.write("Selected Data")
st.write(data)

# ADF test for stationarity
st.header('Is data stationary?')
if not data[column].empty:
    st.write(adfuller(data[column])[1] < 0.05)
else:
    st.write("Data is empty")

# Convert date column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# SARIMAX Model
model = sm.tsa.statespace.SARIMAX(data[column], order=(2, 1, 2))
model = model.fit()

# Forecasting
st.write("<p style='color:green; font-size: 50px; font-weight:bold;'>Forecasting the data</p>", unsafe_allow_html=True)
forecast_period = st.number_input("Select the number of days to forecast", 1, 365, 7)
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period-1)
predictions = predictions.predicted_mean

# Set index and format predictions
predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index)
predictions.reset_index(drop=True, inplace=True)
st.write("## Predictions", predictions)
st.write("## Actual Data", data)

# Plotting actual vs predicted data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
st.plotly_chart(fig)

# Calculate risk or profit
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
    

# Nifty50 visualization
#st.header('Nifty50 Index Trend')

# Fetch Nifty50 data from Yahoo Finance
#nifty50_ticker = '^NSEI'
#nifty50_data = yf.download(nifty50_ticker, start=start_date, end=end_date)

# Adding a Date column and resetting index
#nifty50_data.insert(0, "Date", nifty50_data.index, True)
#nifty50_data.reset_index(drop=True, inplace=True)

# Display Nifty50 data
#st.write(f"--- Nifty50 Data from {start_date} to {end_date} ---")
#st.write(nifty50_data)

# Plot Nifty50 Opening and Closing Price trend
#st.subheader('Nifty50 Opening and Closing Price Trend')
#fig_nifty50 = go.Figure()

# Add Opening price to the plot
#fig_nifty50.add_trace(go.Scatter(
#    x=nifty50_data['Date'], y=nifty50_data['Open'], mode='lines', name='Opening Price', line=dict(color='blue')
#))

# Add Closing price to the plot
#fig_nifty50.add_trace(go.Scatter(
#    x=nifty50_data['Date'], y=nifty50_data['Close'], mode='lines', name='Closing Price', line=dict(color='red')
#))

#fig_nifty50.update_layout(
#    title='Nifty50 Opening and Closing Price Trend',
#    xaxis_title='Date',
#    yaxis_title='Price',
#    width=1000,
#    height=400
#)
#st.plotly_chart(fig_nifty50)

# Show percentage change over the period for both Open and Close
#nifty50_data['Open Change (%)'] = nifty50_data['Open'].pct_change() * 100
#nifty50_data['Close Change (%)'] = nifty50_data['Close'].pct_change() * 100
#st.write('--- Nifty50 Opening and Closing Percentage Change Over Time ---')
#st.write(nifty50_data[['Date', 'Open', 'Close', 'Open Change (%)', 'Close Change (%)']])

# Plot Nifty50 Percentage Change for both Open and Close prices
#st.subheader('Nifty50 Opening and Closing Price Percentage Change')
#fig_nifty50_change = go.Figure()

# Add Open Change to the plot
#fig_nifty50_change.add_trace(go.Scatter(
#    x=nifty50_data['Date'], y=nifty50_data['Open Change (%)'], mode='lines', name='Open Change (%)', line=dict(color='green')
#))

# Add Close Change to the plot
#fig_nifty50_change.add_trace(go.Scatter(
#    x=nifty50_data['Date'], y=nifty50_data['Close Change (%)'], mode='lines', name='Close Change (%)', line=dict(color='orange')
#))

#fig_nifty50_change.update_layout(
#    title='Nifty50 Opening and Closing Price Percentage Change',
#    xaxis_title='Date',
#    yaxis_title='Percentage Change (%)',
#    width=1000,
#    height=400
#)
#st.plotly_chart(fig_nifty50_change)


# Suggest top 5 similar stocks
#st.header('Top 5 Similar Stocks')

# Filter stocks with the same sector and industry
#similar_stocks = sp500_table[(sp500_table['GICS Sector'] == sector) & (sp500_table['Symbol'] != ticker)].head(5)
#similar_stocks = similar_stocks[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

# Display similar stocks
#st.write(similar_stocks)

# News Section - Latest Stock Market News
#st.header('Latest Stock Market News')

#api_key = 'e13f2a22dd724be6b5d4f5d782b0cd40'  # Your News API key
#news_url = f'https://newsapi.org/v2/everything?q=stock%20market&sortBy=publishedAt&apiKey={api_key}'

#try:
#    response = requests.get(news_url)
#    response.raise_for_status()  # Raise an error for bad responses
#    news_data = response.json()

#    if news_data['status'] == 'ok':
#        articles = news_data['articles'][:5]  # Get the top 5 latest news articles
#        for article in articles:
#            st.subheader(article['title'])
#            st.write(f"Source: {article['source']['name']}")
#            st.write(article['description'])
#            st.write(f"[Read more]({article['url']})")
#           st.write("---")
#    else:
#        st.write("Failed to fetch news.")
#except requests.exceptions.RequestException as e:
#    st.write(f"An error occurred: {e}")
