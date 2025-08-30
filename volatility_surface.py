##==============================================================================
## Implied Volatility Suface
## Anthony Le
## July 19, 2025
## IV_surface.py
##  
## Description:
## This script calculates and visualizes the implied volatility surface for 
## options using the Black-Scholes model. It allows users to input parameters
## such as ticker symbol, risk-free rate, dividend yield, time to expiration,
## and strike price range. The surface plot shows how implied volatility varies
## with time to expiration and strike price (or moneyness).
##
## Dependencies:
## - yfinance: For fetching financial data from Yahoo Finance.
## - pandas: For data manipulation and analysis.
## - numpy: For numerical operations.
## - scipy: For scientific computing, including optimization and interpolation.
## - streamlit: For creating the web application interface.
## - plotly: For interactive plotting of the implied volatility surface.
##==============================================================================

## Import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go


##===============================================================================
##===============================================================================
## Function Definitions
## This section contains the function definitions for calculating the call option
## prices using the Black-Scholes formula, calculating implied volatility, 
## retrieving and processing options data.
##===============================================================================

##===============================================================================
## bs_call_price
## This function calculates the Black-Scholes call option price.
## Parameters:
##    S: Spot price of the underlying asset
##    K: Strike price of the option 
##    T: Time to expiration in years
##    r: Risk-free interest rate
##    sigma: Volatility of the underlying asset
##    q: Dividend yield (default is 0)
## Returns:
##    Call option price
##===============================================================================
def bs_call_price(S, K, T, r, sigma, q=0):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * np.exp(-q * T) * norm.cdf(d1) -
                  K * np.exp(-r * T) * norm.cdf(d2))
    return call_price


##===============================================================================
## implied_volatility
## This function calculates the implied volatility using the Brent's method.
## Parameters:
##    price: Market price of the option
##    S: Spot price of the underlying asset 
##    K: Strike price of the option
##    T: Time to expiration in years
##    r: Risk-free interest rate
##    q: Dividend yield (default is 0)
## Returns:
##    Implied volatility or NaN if calculation fails 
##===============================================================================
def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan

    def objective_function(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price

    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        implied_vol = np.nan

    return implied_vol


##===============================================================================
## get_dividend_yield
## This function retrieves the dividend yield from Yahoo Finance.
## Parameters:
##    ticker_symbol: Ticker symbol of the stock
## Returns:
##    Dividend yield in decimal or 0.0 if not available
## Note: The dividend yield is divided by 100 to convert it to decimal.
##================================================================================
def get_dividend_yield(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    try:
        dividend_yield = ticker.info['dividendYield']
        if dividend_yield is not None:
            return dividend_yield / 100  # Convert to decimal
        else:
            return 0.0
    except Exception as e:
        return 0.0


##================================================================================
## get_spot_price 
## This function retrieves the spot price of the underlying asset.
## Parameters:
##    ticker_symbol: Ticker object of the stock
## Returns:
##    Spot price of the underlying asset or NaN if not available
##=================================================================================
def get_spot_price(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    if not ticker:
        st.error(f'Invalid ticker symbol: {ticker_symbol}. Please try again.')
        st.stop()
        return np.nan

    try:
        spot_price = ticker.info['regularMarketPrice']
        if pd.isna(spot_price):
            st.error(f'Failed to retrieve spot price data for {ticker_symbol}.')
            st.stop()
            return np.nan
        return spot_price
    
    except Exception as e:
        st.error(f'An error occurred while fetching spot price data: {e}')
        st.stop()
        return np.nan


##================================================================================
## get_valid_expiration_dates
## This function fetches the available expiration dates for the specified ticker.
## It filters the dates based on user input for time to expiration.
## Parameters:
##    ticker_symbol: Ticker object of the stock
##    time_min: Minimum time to expiration in years
##    time_max: Maximum time to expiration in years
## Returns:
##    List of valid expiration dates or an empty list if none found
##=================================================================================
def get_valid_expiration_dates(ticker, time_min, time_max):
    today = pd.Timestamp('today').normalize()
    
    try:
        expirations = ticker.options
    except Exception as e:
        st.error(f'Error fetching options for {ticker_symbol}: {e}')
        return []
    
    if not expirations:
        st.error(f'No available option expiration dates for {ticker_symbol}.')
        return []

    # Filter expiration dates based on user input
    valid_dates = [
        pd.Timestamp(date) for date in expirations
        if (pd.Timestamp(date) >= today + timedelta(days=365 * time_min)) and
           (pd.Timestamp(date) <= today + timedelta(days=365 * time_max))
    ]

    return valid_dates


##================================================================================
## fetch_option_data
## This function retrieves the option chain data for the specified ticker symbol
## and expiration dates. It filters the options based on bid and ask prices,
## and calculates the mid price for each option.
## Parameters:
##    ticker_symbol: Ticker object of the stock
##    exp_dates: List of expiration dates to fetch data for
## Returns:
##    DataFrame containing option data with expiration date, strike, and mid price.
##=================================================================================
def fetch_option_data(ticker, exp_dates):
    option_data = []

    # Loop through each expiration date
    for exp_date in exp_dates:
        # Fetch option chain for the expiration date
        try:
            opt_chain = ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
            calls = opt_chain.calls
        except Exception as e:
            st.warning(f'Failed to fetch option chain for {exp_date.date()}: {e}')
            continue

        # Filter out options with zero bid or ask prices
        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]

        # Calculate mid price and store relevant data
        for index, row in calls.iterrows():
            strike = row['strike']
            mid_price = (row['bid'] + row['ask']) / 2

            option_data.append({
                'expirationDate': exp_date,
                'strike': strike,
                'mid': mid_price
            })
  
    return pd.DataFrame(option_data)


##================================================================================
## process_options_data
## This function processes the options data by filtering based on strike price
## range and calculating time to expiration (in years).
## Parameters:
##    option_df: DataFrame containing option data
##    spot_price: Spot price of the underlying asset
##    min_strike_pct: Minimum strike price percentage of spot price
##    max_strike_pct: Maximum strike price percentage of spot price
## Returns:
##    DataFrame containing options data with calculated implied volatility.
##=================================================================================
def process_options_data(option_df, spot_price, min_strike_pct, max_strike_pct):
    # Filter options based on strike price range
    option_df = option_df[
        (option_df['strike'] >= spot_price * (min_strike_pct / 100)) &
        (option_df['strike'] <= spot_price * (max_strike_pct / 100))    
    ]

    # Calculate time to expiration in years
    today = pd.Timestamp('today').normalize()
    option_df['daysToExpiration'] = (option_df['expirationDate'] - today).dt.days
    option_df['timeToExpiration'] = option_df['daysToExpiration'] / 365
    option_df.reset_index(drop=True, inplace=True)

    return option_df


##================================================================================
## finalize_options_data
## This function finalizes the options data by calculating the implied volatility
## for each option using the Black-Scholes model, dropping NaN values, and sorting
## the DataFrame by strike price.
## Parameters:
##    options_df: DataFrame containing options data
##    spot_price: Spot price of the underlying asset
##    risk_free_rate: Risk-free interest rate
##    dividend_yield: Dividend yield of the underlying asset
## Returns:
##    DataFrame containing options data with calculated implied volatility.
##=================================================================================
def finalize_options_data(options_df, spot_price, risk_free_rate, dividend_yield):
    # Calculate implied volatility for each option
    with st.spinner('Calculating implied volatility...'):
        options_df['impliedVolatility'] = options_df.apply(
            lambda row: implied_volatility(
                price=row['mid'],
                S=spot_price,
                K=row['strike'],
                T=row['timeToExpiration'],
                r=risk_free_rate,
                q=dividend_yield
            ), axis=1
        )

    # Clean up dataframe
    options_df.dropna(subset=['impliedVolatility'], inplace=True)
    options_df['impliedVolatility'] *= 100  # Convert to percentage
    options_df.sort_values('strike', inplace=True)
    options_df['moneyness'] = options_df['strike'] / spot_price

    return options_df


##===============================================================================
##===============================================================================
## Streamlit Application Setup
## This section sets up the Streamlit application, including the title, sidebar,
## and main content. It allows users to input parameters for the Black-Scholes 
## model and visualizes the implied volatility surface.
##===============================================================================

##===============================================================================
## Application title and description
## This section sets the title and description of the Streamlit application.
##===============================================================================
st.title("📊 Implied Volatility Surface")
linkedin_url = "https://www.linkedin.com/in/anthony-hn-le/"
st.markdown(f'`Created by:` <a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"> <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px; margin-left: 10px;">`Anthony Le`</a>', unsafe_allow_html=True)

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded")

st.write(
    """This application calculates and visualizes the implied volatility surface 
    for options using the Black-Scholes model. You can choose a ticker symbol and 
    set the risk-free rate, dividend yield, time to expiration, and strike price
    range. The surface plot will show how implied volatility varies with time to
    expiration and strike price (or moneyness). """)


##===============================================================================
## Sidebar for user inputs
## This section allows users to input parameters for the Black-Scholes model.
##===============================================================================
with st.sidebar:
    st.title("Model Parameters")
    
    ticker_symbol = st.text_input(
        'Ticker Symbol',
        value='SPY',
        max_chars=5
    ).upper()

    risk_free_rate = st.number_input(
        '''Risk-Free Rate (e.g., 0.0425 for 4.25%)
        (Real-time data from Yahoo Finance by default)''',
        value= yf.Ticker("^IRX").info['regularMarketPrice'] / 100,
        step=0.001,
        format="%.4f"
    )

    dividend_yield = st.number_input(
        '''Dividend Yield (e.g., 0.0615 for 6.15%)
        (Real-time data from Yahoo Finance by default)''',
        value= get_dividend_yield(ticker_symbol),
        format="%.4f"
    )

    time_min, time_max = st.slider(
        'Time to Expiration (Years)',
        min_value=0.00,
        max_value=3.0,
        value=(0.01, 1.0),
        step=0.01,
        format="%.2f"
    )

    min_strike_pct, max_strike_pct = st.slider(
        'Strike Price Range (% of Spot Price)',
        min_value=50,
        max_value=200,
        value=(80, 120),
        step=1,
        format="%.0f"
    )
    
    y_axis_option = st.selectbox(
        'Select Y-axis:',
        ('Strike Price ($)', 'Moneyness')
    )


##================================================================================
## Reporting 
## This section displays the ticker information and other basic parameters, 
## such as spot price, dividend yield, and risk-free rate.
##=================================================================================

ticker = yf.Ticker(ticker_symbol)
spot_price = get_spot_price(ticker_symbol)

## Report the ticker, spot price, dividend yield and risk-free rate
col1, col2, col3, col4 = st.columns(4, border=True)

with col1:
    st.subheader('Ticker', divider=True)
    st.markdown(f"**{ticker_symbol}**")
with col2:
    st.subheader('Spot Price', divider=True)
    st.markdown(f"**${spot_price:.2f}**")
with col3:
    st.subheader('Dividend Yield', divider=True)
    st.markdown(f"**{dividend_yield:.2%}**")
with col4:
    st.subheader('Risk-Free Rate', divider=True)
    st.markdown(f"**{risk_free_rate:.2%}**")

st.info("""
    Note: By default, the risk-free rate is set to the current yield of the
    13-week Treasury bill (IRX), while the dividend yield is latest yield of the
    underlying asset. Both values are automatically fetched from Yahoo Finance. """)

st.divider() 

##===============================================================================
## Main content
## This section retrieves the options data, processes it, and visualizes the
## implied volatility surface using Plotly.
##===============================================================================

# Retrieve and process options data to calculate the implied volatility surface
exp_dates = get_valid_expiration_dates(ticker, time_min, time_max)
if not exp_dates:
    st.error('Please check your inputs and try again.')
    st.stop()

option_data = fetch_option_data(ticker, exp_dates)

options_df = process_options_data(option_data, spot_price, min_strike_pct, max_strike_pct)
if options_df.empty:
    st.error('No option data available at this time due to market hours. Please try again later.')
    st.stop()

options_df = finalize_options_data(options_df, spot_price, risk_free_rate, dividend_yield)


## Set up the plotly surface chart
if y_axis_option == 'Strike Price ($)':
    Y = options_df['strike'].values
    y_label = 'Strike Price ($)'
else:
    Y = options_df['moneyness'].values
    y_label = 'Moneyness (Strike / Spot)'

X = options_df['timeToExpiration'].values
Z = options_df['impliedVolatility'].values

ti = np.linspace(X.min(), X.max(), 50)
ki = np.linspace(Y.min(), Y.max(), 50)
T, K = np.meshgrid(ti, ki)

Zi = griddata((X, Y), Z, (T, K), method='linear')

Zi = np.ma.array(Zi, mask=np.isnan(Zi))

fig = go.Figure(data=[go.Surface(
    x=T, y=K, z=Zi,
    colorscale='Viridis',
    colorbar_title='Implied Volatility (%)'
)])

fig.update_layout(
    title=f'Implied Volatility Surface for {ticker_symbol} Options',
    scene=dict(
        xaxis_title='Time to Expiration (years)',
        yaxis_title=y_label,
        zaxis_title='Implied Volatility (%)'
    ),
    autosize=False,
    width=1200,
    height=700,
    margin=dict(l=30, r=30, b=60, t=30)
)

st.plotly_chart(fig)

st.divider() 

st.write("""
    **Acknowledgements**: This project was inspired by the recommendation of 
    [Coding Jesus](https://www.youtube.com/@CodingJesus) on YouTube and the work of
    [Mateusz Jastrzębski](https://www.linkedin.com/in/mateusz-jastrz%C4%99bski-8a2622264/).
    
    **Disclaimer**: This application is for educational purposes only and does not
    constitute financial advice. The implied volatility surface is calculated using
    the Black-Scholes model, which has limitations and assumptions that may not hold
    true in all market conditions. Always conduct your own research and consult with
    a financial advisor before making investment decisions.
         
    **Feedback**: If you have any suggestions or feedback, please feel free to
    reach out to me on [LinkedIn](https://www.linkedin.com/in/anthony-hn-le/).
    Thank you for using this application! I hope you find it useful in your options
    trading and analysis. Happy trading! 😊
""")


##===============================================================================
## End of the application
##===============================================================================