from decimal import Decimal
import streamlit as st
import mysql.connector
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests 
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from cryptography.fernet import Fernet

# Define a single static key for encryption/decryption (Store this securely in real-world applications)
key = b'DPQ1H9pe1Xp0dqQg9xhB68Z1ubqOY0UKuueIEJaJMQ8='  # Example key, must be exactly 32 bytes base64 encoded
cipher_suite = Fernet(key)

# Database Configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "12345",
    "database": "stock_market"  # Use the same database in both login and registration
}

# Email Configuration
smtp_server = 'smtp.gmail.com'
smtp_port = 587
sender_email = 'kudiyayash31@gmail.com'
sender_password = 'blxx sdpn ahwt mifa'

otp = None

def connect_to_db():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        st.error(f"Database connection error: {err}")
        return None

def send_otp(username, email):
    otp = random.randint(100000, 999999)
    st.session_state.otp = otp  # Store OTP in session state to persist it
    message = f"Your OTP for login is {otp}"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = 'OTP for Login'
    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, msg.as_string())
        server.quit()
        st.success(f"OTP sent to {email}. Please check your inbox.")
    except Exception as e:
        st.error(f"Failed to send OTP: {e}")

def verify_otp(username, input_otp):
    if 'otp' in st.session_state:
        if str(st.session_state.otp) == str(input_otp):  # Compare as strings
            # After OTP is verified, redirect the user to the dashboard
            conn = connect_to_db()
            if conn:
                cursor = conn.cursor()
                query = "SELECT * FROM INFO WHERE username = %s"
                cursor.execute(query, (username,))
                user = cursor.fetchone()
                conn.close()
                if user:
                    st.session_state["user_logged_in"] = True
                    st.session_state["username"] = username
                    st.success("OTP verified successfully! Redirecting to the dashboard...")

                    # Set logged_in session state
                    st.session_state.logged_in = True
                    st.rerun()  # This will trigger a page rerun and load the app page

                else:
                    st.error("User not found. Please register first.")
            else:
                st.error("Database connection error.")
        else:
            st.error("Invalid OTP. Please try again.")
    else:
        st.error("OTP not generated. Please request a new OTP.")

def encrypt_password(password):
    """Encrypt the password using Fernet."""
    return cipher_suite.encrypt(password.encode())

def decrypt_password(encrypted_password):
    """Decrypt the password using Fernet."""
    return cipher_suite.decrypt(encrypted_password).decode()

def login_user():
    st.title("Login")
    
    # Initialize session state variables if not already set
    if 'otp_sent' not in st.session_state:
        st.session_state.otp_sent = False
    if 'otp_verified' not in st.session_state:
        st.session_state.otp_verified = False
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Username input (always visible)
    username = st.text_input("Username")
    
    # Password input (always visible)
    password = st.text_input("Password", type="password")
    
    # Email input (always visible)
    email = st.text_input("Email")
    
    # Send OTP button
    if st.button("Send OTP") and not st.session_state.otp_sent:
        if username and email:
            # Send OTP
            send_otp(username, email)
            st.session_state.otp_sent = True
        else:
            st.error("Please enter both username and email.")
    
    # OTP input field (visible only after OTP is sent)
    if st.session_state.otp_sent and not st.session_state.logged_in:
        otp_input = st.text_input("Enter OTP")
        
        # Verify OTP button (visible only after OTP is sent)
        if st.button("Verify OTP"):
            if otp_input:
                # Use the existing verify_otp function
                verify_otp(username, otp_input)
            else:
                st.error("Please enter the OTP.")
    
    # If user is logged in through OTP verification, show success
    if st.session_state.get("user_logged_in", False):
        st.success("Login Successful!")
        # After successful login, redirect to the dashboard
        dasboard()

def register_user():
    st.title("Register New User")
    st.subheader("Create your account")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    email = st.text_input("Email")

    if st.button("Register"):
        if username and password and email:
            conn = connect_to_db()
            if conn:
                cursor = conn.cursor()
                cursor.execute(""" 
                    CREATE TABLE IF NOT EXISTS INFO (
                        ID INT AUTO_INCREMENT PRIMARY KEY,
                        Email VARCHAR(100) UNIQUE,
                        Username VARCHAR(50),
                        Password VARCHAR(100)
                    )
                """)
                encrypted_password = encrypt_password(password)
                try:
                    cursor.execute("INSERT INTO INFO (Email, Username, Password) VALUES (%s, %s, %s)", 
                                   (email, username, encrypted_password.decode()))
                    conn.commit()
                    st.success("User registered successfully!")
                    st.rerun()
                except mysql.connector.IntegrityError:
                    st.error("Username or email already registered.")
                conn.close()
        else:
            st.error("Please fill all fields.")

def fetch_current_price(ticker):
    """Fetch the current price of the stock using yfinance."""
    data = yf.download(ticker, period='1d', group_by='ticker', auto_adjust=True)
    return data[ticker]['Close'][-1] if not data.empty else None

def buy_stock(ticker, current_price):
    username = st.session_state["username"]  # Get the logged-in username
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO user_stocks (username, ticker, purchase_price) VALUES (%s, %s, %s)",
                           (username, ticker, Decimal(current_price)))
            conn.commit()
            st.success(f"Stock {ticker} bought at price {current_price}.")
        except mysql.connector.Error as err:
            st.error(f"Error buying stock: {err}")
        finally:
            cursor.close()
            conn.close()
        show_user_stocks()

def sell_stock(ticker):
    username = st.session_state["username"]  # Get the logged-in username
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        try:
            # Fetch the purchase price
            cursor.execute("SELECT purchase_price FROM user_stocks WHERE username = %s AND ticker = %s", (username, ticker))
            purchase_record = cursor.fetchone()
            if purchase_record:
                purchase_price = purchase_record[0]
                current_price = fetch_current_price(ticker)
                
                if current_price is not None:  # Check if current_price is valid
                    current_price = Decimal(current_price)  # Convert to Decimal
                    profit_or_loss = (current_price - Decimal(purchase_price)) / Decimal(purchase_price) * 100
                    
                    # Delete the stock from user's stocks
                    cursor.execute("DELETE FROM user_stocks WHERE username = %s AND ticker = %s", (username, ticker))
                    conn.commit()
                    
                    if profit_or_loss > 0:
                        st.success(f"Sold stock {ticker} at profit of {profit_or_loss:.2f}%.")
                    else:
                        st.success(f"Sold stock {ticker} at loss of {abs(profit_or_loss):.2f}%.")
                else:
                    st.error("Current price could not be fetched. Please try again later.")
            else:
                st.error("Stock not found in your portfolio.")
        except mysql.connector.Error as err:
            st.error(f"Error selling stock: {err}")
        finally:
            cursor.close()
            conn.close()
        show_user_stocks()
        
def show_user_stocks():
    username = st.session_state["username"]  # Get the logged-in username
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, purchase_price, purchase_date FROM user_stocks WHERE username = %s", (username,))
        stocks = cursor.fetchall()
        conn.close()
        
        if stocks:
            st.subheader("Your Stocks")
            for stock in stocks:
                st.write(f"Ticker: {stock[0]}, Purchase Price: {stock[1]}, Purchase Date: {stock[2]}")
        else:
            st.write("You have no stocks bought yet.")

def dasboard():
    # App name
    app_name = 'Stock Market Analytics'
    st.title(app_name)
    st.subheader('This application is created to predict the stock market price of the selected company.')

    # Display an image
    st.image("https://akm-img-a-in.tosshub.com/indiatoday/images/story/202401/top-stock-picks-for-today-the-market-veteran-also-liked-bharat-dynamics-ltd-164112329-16x9.jpg?VersionId=Xx6nIyHvkXKxieGa_KqwFIv.GCZWvK_9&size=690:388")

    # Function to fetch top gainers and losers with company name and sector
    def fetch_top_gainers_losers():
        # Fetch S&P 500 company tickers
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="12345",
            database="stock_market"
        )

        query = "SELECT Symbol, GICS_Sector, Security FROM sp500_companies"
        cursor = db_connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()

        tickers = [row[0] for row in results]
        sectors = {row[0]: row[1] for row in results}
        company_names = {row[0]: row[2] for row in results}

        cursor.close()
        db_connection.close()

    
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

    # Sidebar header
    st.sidebar.header("Select Input Parameters Below")

    # Adding a button to refresh data
    refresh_button = st.sidebar.button("Refresh Data")

    # Taking start date and end date as input from user
    start_date = st.sidebar.date_input('Start Date', date.today() - timedelta(days=365))
    end_date = st.sidebar.date_input('End Date', date.today())

    # Fetching S&P 500 company tickers
    # Connect to MySQL database
    db_connection = mysql.connector.connect(
        host="localhost",      # Replace with your host
        user="root",      # Replace with your username
        password="12345",  # Replace with your password
        database="stock_market"
    )
    

    # Fetch tickers, sectors, and company names from sp500_companies table
    query = "SELECT Symbol, Security, GICS_Sector FROM sp500_companies"
    cursor = db_connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()

    # Organize the results into lists/dictionaries
    ticker_list = [row[0] for row in results]  # List of Symbols
    company_names = {row[0]: row[1] for row in results}  # Symbol -> Company Name
    sectors = {row[0]: row[2] for row in results}  # Symbol -> GICS Sector

    ticker = st.sidebar.selectbox('Select the company', ticker_list)
    cursor.close()
    db_connection.close()


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
    try:
        # Use the database connection to fetch sector and industry information
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="12345",
            database="stock_market"
        )
    
        cursor = db_connection.cursor(dictionary=True)
        query = "SELECT GICS_Sector, GICS_Sub_Industry FROM sp500_companies WHERE Symbol = %s"
        cursor.execute(query, (ticker,))
    
        company_info = cursor.fetchone()
    
        if company_info:
            sector = company_info['GICS_Sector']
            industry = company_info['GICS_Sub_Industry']
        
            # Display sector and industry with styled Streamlit markdown
            st.markdown(f"""
            <div style='display: flex; flex-direction: column; gap: 10px;'>
                <p style='color:red; font-size: 20px; font-weight:bold;'>
                    Entered Stock Sector: 
                    <span style='color:green'>{sector}</span>
                </p>
                <p style='color:red; font-size: 20px; font-weight:bold;'>
                    Entered Stock Sub-Industry: 
                    <span style='color:green'>{industry}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"No information found for ticker {ticker}")

    except mysql.connector.Error as err:
        st.error(f"Database connection error: {err}")

    finally:
        # Ensure the database connection is closed
        if 'cursor' in locals():
            cursor.close()
        if 'db_connection' in locals():
            db_connection.close()
        
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

    #Flatten the multi-level columns if necessary
    data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in data.columns]

    # Plotting the data
    st.header('Visualizing the data')
    st.subheader('Plotting the data')
    st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")

    # Ensure 'Date' is a column, not an index
    if 'Date' not in data.columns:
        data.reset_index(inplace=True)  # Reset index if 'Date' is currently the index
        data = data.rename(columns={'index': 'Date'})
        

    # Select the column for forecasting, excluding 'Date' and any column starting with 'Date_'
    column_options = [col for col in data.columns if col != 'Date' and not col.startswith('Date_')]
    column = st.selectbox('Select the column for forecasting', column_options)
    plot_data = data[[ column]]  # Now we have the 'Date' column and the selected column

    st.write("Selected Data")
    st.write(plot_data)

    # Now use `column` in the plot function
    fig = px.line(data, x='Date', y=column, width=2500, height=500)
    st.plotly_chart(fig)

    # ADF test for stationarity
    st.header('Is data stationary?')
    if not data[column].empty and data[column].isnull().sum() < len(data):
        st.write(adfuller(data[column])[1] < 0.05)
    else:
        st.write("Data is empty or contains only null values.")

    # Convert date column to datetime and set as index
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data.set_index('Date', inplace=True)

    # SARIMAX Model
    model = sm.tsa.statespace.SARIMAX(data[column], order=(1, 1, 1))    
    model = model.fit()

    # Forecasting
    st.write("<p style='color:green; font-size: 50px; font-weight:bold;'>Forecasting the data</p>", unsafe_allow_html=True)
    forecast_period = st.number_input("Select the number of days to forecast", 1, 365, 7)
    end_date_selected = pd.to_datetime(end_date)
    start_forecast_date = end_date_selected + pd.Timedelta(days=1)
    
    predictions = model.get_forecast(steps=forecast_period)
    predictions = predictions.predicted_mean

    # Set index and format predictions
    forecast_dates = pd.date_range(start=start_forecast_date, periods=forecast_period, freq='D')
    predictions.index = forecast_dates
    
    # Convert Series to DataFrame before using insert
    predictions_df = predictions.to_frame(name='predicted_mean')
    predictions_df.insert(0, 'Date', predictions_df.index)

    # Reset the index (if needed) and display predictions
    predictions_df.reset_index(drop=True, inplace=True)
    st.write("## Predictions", predictions_df)

    # Plotting actual vs predicted data with zig-zag effect for actual data
    fig = go.Figure()

    # Add predicted data trace
    fig.add_trace(
        go.Scatter(
            x=predictions_df['Date'],
            y=predictions_df['predicted_mean'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red')
        )
    )

    # Adjust layout
    fig.update_layout(
        title='Predicted',
     xaxis_title='Date',
        yaxis_title='Price',
        width=1000,
        height=400
    )

    # Display the plot
    st.plotly_chart(fig)

    # Calculate risk or profit
    actual_last_price = data[column].iloc[-1]
   # Ensure predictions is converted to DataFrame
    predictions_df = predictions.to_frame(name='predicted_mean')

    # Access the 'predicted_mean' column from predictions_df
    predicted_last_price = predictions_df["predicted_mean"].iloc[-1]


    if pd.notna(actual_last_price) and pd.notna(predicted_last_price):
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
    else:
        st.write("Unable to calculate risk or profit due to missing data.")

    if st.session_state.get("user_logged_in", False):
        current_price = data[column].iloc[-1]  # Get the current price of the selected stock
        if st.button("Buy Stock"):
            buy_stock(ticker, current_price)
        
        if st.button("Sell Stock"):
            sell_stock(ticker)

    # Call to show stocks after actions
    show_user_stocks()
    
    # Suggest top 5 similar stocks
    st.header('Top 5 Similar Stocks')

    # Connect to database to fetch similar stocks
    try:
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="12345",
            database="stock_market"
        )
    
        # Prepare SQL query to find similar stocks
        query = """
        SELECT Symbol, Security, GICS_Sector, GICS_Sub_Industry 
        FROM sp500_companies 
        WHERE GICS_Sector = %s 
        AND Symbol != %s 
        LIMIT 5
        """
    
        cursor = db_connection.cursor(dictionary=True)
        cursor.execute(query, (sector, ticker))
    
        # Fetch similar stocks
        similar_stocks = pd.DataFrame(cursor.fetchall())
    
        # Display similar stocks if found
        if not similar_stocks.empty:
            st.write(similar_stocks[['Symbol', 'Security', 'GICS_Sector', 'GICS_Sub_Industry']])
        else:
            st.write("No similar stocks found in the same sector.")

    
    except mysql.connector.Error as err:
        st.error(f"Database error: {err}")
    finally:
        # Ensure connection is closed
        if 'cursor' in locals():
            cursor.close()
        if 'db_connection' in locals():
            db_connection.close()

    # News Section - Latest Stock Market News
    st.header('Latest Stock Market News')

    api_key = 'e13f2a22dd724be6b5d4f5d782b0cd40'  # Your News API key
    news_url = f'https://newsapi.org/v2/everything?q=stock%20market&sortBy=publishedAt&apiKey={api_key}'

    try:
        response = requests.get(news_url)
        response.raise_for_status()  # Raise an error for bad responses
        news_data = response.json()

        if news_data['status'] == 'ok':
            articles = news_data['articles'][:5]  # Get the top 5 latest news articles
            for article in articles:
                st.subheader(article['title'])
                st.write(f"Source: {article['source']['name']}")
                st.write(article['description'])
                st.write(f"[Read more]({article['url']})")
                st.write("---")
        else:
            st.write("Failed to fetch news.")
    except requests.exceptions.RequestException as e:
        st.write(f"An error occurred: {e}")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        # If logged in, show the dashboard page only
         dasboard()
    else:
        st.sidebar.title("Navigation")
        choice = st.sidebar.radio("Choose Action", ["Login", "Register"])

        if choice == "Login":
            login_user()
        elif choice == "Register":
            register_user()
 
if __name__ == "__main__":
    main()
