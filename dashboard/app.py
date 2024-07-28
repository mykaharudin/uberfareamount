import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io

# Function to load data
@st.cache_data
def load_data():
    url = 'data/uber.csv'
    df = pd.read_csv(url)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    return df

# Function to calculate distance using haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance

# Function to preprocess data
def preprocess_data(df):
    df = df.rename(columns={"Unnamed: 0": "Id"})
    df = df.drop(columns=['key'])
    df = df.dropna()
    df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
    df['month'] = df['pickup_datetime'].dt.month_name()
    df['day'] = df['pickup_datetime'].dt.day_name()
    df['rush_hour'] = df['pickup_datetime'].dt.hour
    df.loc[df['day'].isin(['Sunday']), 'rush_hour'] = 1

    def rush_hourizer(hour):
        if 6 <= hour['rush_hour'] < 10:
            return 1
        elif 16 <= hour['rush_hour'] < 20:
            return 1
        else:
            return 0

    df.loc[(df.day != 'Sunday'), 'rush_hour'] = df.apply(rush_hourizer, axis=1).astype('int32')
    df1 = df.drop(['Id', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'day', 'month'], axis=1)
    df1['rush_hour'] = df1['rush_hour'].astype(float)
    df1 = df1[df1['distance_km'] != 0].reindex()
    return df, df1

# Function to train model
def train_model(df1):
    X = df1.drop(columns=['fare_amount'])
    y = df1[['fare_amount']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    r_sq = lr.score(X_train_scaled, y_train)
    y_pred_train = lr.predict(X_train_scaled)
    train_metrics = {
        'r_sq': r_sq,
        'mae': mean_absolute_error(y_train, y_pred_train),
        'mse': mean_squared_error(y_train, y_pred_train),
        'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
    }

    X_test_scaled = scaler.transform(X_test)
    r_sq_test = lr.score(X_test_scaled, y_test)
    y_pred_test = lr.predict(X_test_scaled)
    test_metrics = {
        'r_sq': r_sq_test,
        'mae': mean_absolute_error(y_test, y_pred_test),
        'mse': mean_squared_error(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
    }

    return lr, scaler, train_metrics, test_metrics

# Load data
df = load_data()
df, df1 = preprocess_data(df)
lr, scaler, train_metrics, test_metrics = train_model(df1)

# Streamlit app layout
st.title('🚕 Analisis Tarif Uber 🚕')

st.sidebar.title('Navigasi')
page = st.sidebar.selectbox("Pilih Halaman", ["Project Overview", "Data Exploration", "Modeling & Evaluation", "Predict Fare"])

# Project Overview
if page == "Project Overview":
    st.subheader('📊 Project Overview')
    st.markdown("""
    Selamat datang di analisis tarif Uber! Proyek ini bertujuan untuk memahami dinamika tarif taksi Uber melalui eksplorasi data dan penerapan model linier, memberikan wawasan berharga bagi strategi bisnis Uber.
    """)
    st.write("### Data Overview")
    st.write(df.head())
    st.write("### Informasi Data")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

# Data Exploration
elif page == "Data Exploration":
    st.subheader('🔍 Data Exploration')
    st.write("### Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)
    st.write("### Missing Values After Cleaning")
    st.write(df.isnull().sum())

    st.write("### Descriptive Statistics")
    st.write(df.describe())

    st.write("### Histogram Fare Amount")
    plt.figure(figsize=(7,3))
    ax = sns.histplot(df['fare_amount'], bins=range(0, 100, 5))
    ax.set_xticks(range(0, 100, 5))
    ax.set_xticklabels(range(0, 100, 5))
    plt.title('Histogram Fare Amount')
    st.pyplot(plt)

    st.write("### Histogram Passenger Count")
    plt.figure(figsize=(7,3))
    ax = sns.histplot(df['passenger_count'], bins=range(0, 20, 2))
    ax.set_xticks(range(0, 20, 2))
    ax.set_xticklabels(range(0, 20, 2))
    plt.title('Histogram Passenger Count')
    st.pyplot(plt)

    # Visualisasi Distribusi Tarif Perjalanan
    st.write("### 💰 Distribusi Tarif Perjalanan")
    mean_fares_by_passenger_count = df.groupby(['passenger_count'])[['fare_amount']].mean()
    st.write(mean_fares_by_passenger_count)

    data = mean_fares_by_passenger_count.tail(-1)
    pal = sns.color_palette("Greens_d", len(data))
    rank = data['fare_amount'].argsort().argsort()
    plt.figure(figsize=(12,7))
    ax = sns.barplot(x=data.index, y=data['fare_amount'], palette=np.array(pal[::-1])[rank])
    ax.axhline(df['fare_amount'].mean(), ls='--', color='red', label='global mean')
    ax.legend()
    plt.title('Average Fare Amount by Passenger Count')
    st.pyplot(plt)

    st.write("### Correlation Heatmap")
    plt.figure(figsize=(6,4))
    sns.heatmap(df1.corr(method='pearson'), annot=True, cmap='Reds')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

# Modeling & Evaluation
elif page == "Modeling & Evaluation":
    st.subheader('🔍 Modeling & Evaluation')

    st.write('### Linear Regression Model:')
    # st.write(f'Coefficient of determination (R²): {train_metrics["r_sq"]:.2f}')
    st.write(f'MAE: {train_metrics["mae"]:.2f}')
    st.write(f'MSE: {train_metrics["mse"]:.2f}')
    st.write(f'RMSE: {train_metrics["rmse"]:.2f}')

    st.write('### Test Data Evaluation:')
    # st.write(f'Coefficient of determination (R²): {test_metrics["r_sq"]:.2f}')
    st.write(f'MAE: {test_metrics["mae"]:.2f}')
    st.write(f'MSE: {test_metrics["mse"]:.2f}')
    st.write(f'RMSE: {test_metrics["rmse"]:.2f}')

# Predict Fare
elif page == "Predict Fare":
    st.subheader('🔮 Predict Fare Amount for New Data')

    passenger_count = st.number_input('Passenger Count', min_value=1, max_value=20, value=1)
    pickup_longitude = st.number_input('Pickup Longitude', value=-73.985428)
    pickup_latitude = st.number_input('Pickup Latitude', value=40.748817)
    dropoff_longitude = st.number_input('Dropoff Longitude', value=-73.985428)
    dropoff_latitude = st.number_input('Dropoff Latitude', value=40.748817)
    pickup_datetime = st.date_input('Pickup Date', value=pd.to_datetime('2012-10-01'))
    pickup_time = st.time_input('Pickup Time', value=pd.to_datetime('2012-10-01 12:00:00').time())

    rush_hour = pickup_time.hour
    if pickup_datetime.weekday() == 6:  # If it's Sunday
        rush_hour = 1
    else:
        if 6 <= rush_hour < 10 or 16 <= rush_hour < 20:
            rush_hour = 1
        else:
            rush_hour = 0

    if st.button('Predict Fare'):
        new_data = pd.DataFrame({
            'passenger_count': [passenger_count],
            'distance_km': [haversine(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)],
            'rush_hour': [rush_hour]
        })

        new_data_scaled = scaler.transform(new_data)
        fare_prediction = lr.predict(new_data_scaled)

        predicted_fare = float(fare_prediction[0])
        st.write(f'Predicted Fare Amount: ${predicted_fare:.2f}')
