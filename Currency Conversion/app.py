
import streamlit as st
from PIL import Image
import pandas as pd
import requests
import json

image = Image.open('logo.jpg')
st.image(image)
st.title('Currency Converter App')
st.sidebar.header('Input Options')

currency_list = ['AUD', 'BGN', 'BRL', 'CAD', 'CHF', 'CNY', 'CZK', 'DKK', 'GBP', 'HKD', 'HRK', 'HUF', 'IDR', 'ILS', 'INR', 'ISK', 'JPY', 'KRW', 'MXN', 'MYR', 'NOK', 'NZD', 'PHP', 'PLN', 'RON', 'RUB', 'SEK', 'SGD', 'THB', 'TRY', 'USD', 'ZAR']
base_price_unit = st.sidebar.selectbox('Select base currency for conversion', currency_list)
symbols_price_unit = st.sidebar.selectbox('Select target currency to convert to', currency_list)
@st.cache
def load_data():
    url = ''.join(['https://api.ratesapi.io/api/latest?base=', base_price_unit, '&symbols=', symbols_price_unit])
    response = requests.get(url)
    data = response.json()
    base_currency = pd.Series( data['base'], name='base_currency')
    rates_df = pd.DataFrame.from_dict( data['rates'].items() )
    rates_df.columns = ['converted_currency', 'price']
    conversion_date = pd.Series( data['date'], name='date' )
    df = pd.concat( [base_currency, rates_df, conversion_date], axis=1 )
    return df

df = load_data()
st.header('Currency conversion')
st.write(df)

