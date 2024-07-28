import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import streamlit as st
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance

st.title('Uber Fare Amount Analysis')

def load_data():
    url = 'data/uber.csv'
    df = pd.read_csv(url)
    return df

df = load_data()

st.subheader('Data Overview')
st.write(df.head())
st.write(df.info())

df = df.rename(columns={"Unnamed: 0": "Id"})
df = df.drop(columns=['key'])

st.subheader('Missing Values')
missing_values = df.isnull().sum()
st.write(missing_values)

df = df.dropna()

st.write(df.isnull().sum())

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

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

df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])

st.write(df.head())

st.subheader('Exploratory Data Analysis')

st.write(df.describe())

plt.figure(figsize=(7,3))
ax = sns.histplot(df['fare_amount'], bins=range(0, 100, 5))
ax.set_xticks(range(0, 100, 5))
ax.set_xticklabels(range(0, 100, 5))
plt.title('Histogram Fare Amount')
st.pyplot(plt)

plt.figure(figsize=(7,3))
ax = sns.histplot(df['passenger_count'], bins=range(0, 20, 2))
ax.set_xticks(range(0, 20, 2))
ax.set_xticklabels(range(0, 20, 2))
plt.title('Histogram Passenger Count')
st.pyplot(plt)

mean_fares_by_passenger_count = df.groupby(['passenger_count']).mean()[['fare_amount']]
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

df['month'] = df['pickup_datetime'].dt.month_name()
df['day'] = df['pickup_datetime'].dt.day_name()

df['rush_hour'] = df['pickup_datetime'].dt.hour
df.loc[df['day'].isin(['Sunday']), 'rush_hour'] = 1

def rush_hourizer(hour):
    if 6 <= hour['rush_hour'] < 10:
        val = 1
    elif 16 <= hour['rush_hour'] < 20:
        val = 1
    else:
        val = 0
    return val

df.loc[(df.day != 'Sunday'), 'rush_hour'] = df.apply(rush_hourizer, axis=1).astype('int32')

df1 = df.drop(['Id', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'day', 'month'], axis=1)
df1['rush_hour'] = df1['rush_hour'].astype(float)

df1 = df1[df1['distance_km'] != 0].reindex()

plt.figure(figsize=(6,4))
sns.heatmap(df1.corr(method='pearson'), annot=True, cmap='Reds')
plt.title('Correlation Heatmap')
st.pyplot(plt)

X = df1.drop(columns=['fare_amount'])
y = df1[['fare_amount']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

r_sq = lr.score(X_train_scaled, y_train)
y_pred_train = lr.predict(X_train_scaled)
st.write('Linear Regression Model:')
st.write('Coefficient of determination:', r_sq)
st.write('R^2:', r2_score(y_train, y_pred_train))
st.write('MAE:', mean_absolute_error(y_train, y_pred_train))
st.write('MSE:', mean_squared_error(y_train, y_pred_train))
st.write('RMSE:', np.sqrt(mean_squared_error(y_train, y_pred_train)))

X_test_scaled = scaler.transform(X_test)

r_sq_test = lr.score(X_test_scaled, y_test)
y_pred_test = lr.predict(X_test_scaled)
st.write('Test Data Evaluation:')
st.write('Coefficient of determination:', r_sq_test)
st.write('R^2:', r2_score(y_test, y_pred_test))
st.write('MAE:', mean_absolute_error(y_test, y_pred_test))
st.write('MSE:', mean_squared_error(y_test, y_pred_test))
st.write('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_test)))

# Add a section for user input and prediction
st.subheader('Predict Fare Amount for New Data')

pickup_latitude = st.number_input('Pickup Latitude', value=40.7614327)
pickup_longitude = st.number_input('Pickup Longitude', value=-73.9798156)
dropoff_latitude = st.number_input('Dropoff Latitude', value=40.6513111)
dropoff_longitude = st.number_input('Dropoff Longitude', value=-73.8803331)
passenger_count = st.number_input('Passenger Count', min_value=1, max_value=6, value=1)
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
    
 # Ensure fare_prediction is a scalar value and convert to float
    predicted_fare = float(fare_prediction[0])
    st.write(f'Predicted Fare Amount: ${predicted_fare:.2f}')