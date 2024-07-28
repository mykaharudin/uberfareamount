import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

import os

"""# Data Ingestion and analysis"""

df = pd.read_csv('https://github.com/mykaharudin/uberfareamount/blob/main/uber.csv')

"""**Data Inspection** <br>

Pertama-tama, kami memeriksa data, termasuk beberapa sampel data, tipe data, penggantian nama kolom, dan identifikasi nilai yang hilang.
"""

df.head()

"""## Columns renaming and droping

Mengganti nama kolom "Unnamed" menjadi "Id" dan hapus kolom "key" karena kolom tersebut merupakan duplikat dari "pickup_datetime.
"""

df = df.rename(columns={"Unnamed: 0": "Id"})
df = df.drop(columns = ['key'])

df.head()

df.info()

"""## Check for missing values

Karena informasi menunjukkan bahwa kedua kolom tersebut memiliki nilai yang hilang, mari kita cari nilai-nilai yang hilang tersebut.
"""

missing_values = df.isnull().sum()
print(missing_values)

"""Karena ini hanya satu baris dan "passenger_count" juga 0, serta key adalah datetime, ini adalah kesalahan dan akan kami hapus."""

df = df.dropna()

# Verify that there are no more missing values (Verifikasi bahwa tidak ada lagi nilai yang hilang)
print(df.isnull().sum())

"""## Data type conversion

Seperti yang kita lihat bahwa "pickup_datetime" adalah tipe object, mari kita konversi menjadi tipe datetime. Pertama, kita akan mengimpor modul datetime.
"""

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['pickup_datetime'].head()

"""### Distance column creation

Karena kami memiliki kolom "latitude" dan "longitude" dan tidak memiliki kolom "distance", mari kita buat kolom tersebut menggunakan [metode haversine](https://en.wikipedia.org/wiki/Haversine_formula.
"""

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers (radius dalam kilometer)

    # Convert degrees to radians (konversi derajat ke radian)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance

# Apply the function to calculate distance for each row (Terapkan fungsi untuk menghitung jarak untuk setiap baris)
df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                              df['dropoff_latitude'], df['dropoff_longitude'])

df.head()

"""## Exploratory Data Analysis (EDA): <br>

Menelusuri beberapa statistik data.
"""

df.describe()

"""Terdapat outlier yang signifikan pada "fare_amount", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", dan "penumpang_count". Misalnya jumlah_tarif minimum adalah -52 (jelas merupakan outlier) dan maks adalah 499 (terlalu tinggi, merupakan outlier).

### Analyzing Data distributions using Visualizations <br>

Membuat beberapa box plots dan histogram untuk mencari sebaran data.

plt.figure(figsize=(7,2))
plt.title('fare amount')
sns.boxplot(data=df, x='fare_amount', fliersize=1)
"""

plt.figure(figsize=(7,3))
ax = sns.histplot(df['fare_amount'],bins=range(0,100,5))
ax.set_xticks(range(0,100,5))
ax.set_xticklabels(range(0,100,5))
plt.title('histogram jumlah harga');

"""Jumlah tarifnya miring ke kanan. dan visualisasi menunjukkan bahwa sebagian besar perjalanan memiliki tarif 5-20 dolar."""

plt.figure(figsize=(7,3))
ax = sns.histplot(df['passenger_count'],bins=range(0,20,2))
ax.set_xticks(range(0,20,2))
ax.set_xticklabels(range(0,20,2))
plt.title('histogram jumlah penumpang');

"""Jadi 2 penumpang per perjalanan adalah mayoritas dan jumlah penumpang mencapai 8.

### Mean fare amount by passenger count

Periksa jumlah tarif dengan jumlah penumpang.
"""

df['passenger_count'].value_counts()

mean_fares_by_passenger_count = df.groupby(['passenger_count']).mean()[['fare_amount']]
mean_fares_by_passenger_count

data = mean_fares_by_passenger_count.tail(-1)
pal = sns.color_palette("Greens_d", len(data))
rank = data['fare_amount'].argsort().argsort()
plt.figure(figsize=(12,7))
ax = sns.barplot(x=data.index,
            y=data['fare_amount'],
            palette=np.array(pal[::-1])[rank])
ax.axhline(df['fare_amount'].mean(), ls='--', color='red', label='global mean')
ax.legend()
plt.title('Rata rata jumlah tarif per jumlah penumpang', fontsize=16);

"""## Feature Engineering: Creating month and day columns by datatime

Buat kolom bulan dan hari untuk memahami data lebih detail.

"""

# Membuat kolom bulan
df['month'] = df['pickup_datetime'].dt.month_name()
# Membuat kolom hari
df['day'] = df['pickup_datetime'].dt.day_name()

"""### Monthly rides

Menganalisis perjalanan berdasarkan bulan.

menyusun ulang bulannya.
"""

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']

monthly_rides = df['month'].value_counts().reindex(index=month_order)
monthly_rides

plt.figure(figsize=(12,7))
ax = sns.barplot(x=monthly_rides.index, y=monthly_rides)
ax.set_xticklabels(month_order)
plt.title('Jumlah perjalanan perbulan', fontsize=16);

"""Perjalanan bulanannya konsisten tetapi pada bulan-bulan musim panas seperti di bulan Juli, Agustus, dan September ada beberapa penurunan.

### Rides per day
"""

daily_rides = df['day'].value_counts()

day_order = ["Monday",'Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
daily_rides.reindex(index=day_order)

plt.figure(figsize=(12,7))
ax = sns.barplot(x=daily_rides.index, y=daily_rides)
ax.set_xticklabels(day_order)
plt.title('Jumlah perjalanan perhari', fontsize=16);

"""Jadi, hari Senin memiliki banyak perjalanan yang jelas karena merupakan awal minggu, sedangkan hari Sabtu dan Minggu mengalami penurunan karena hari libur.

### Revenue per day
"""

df_without_date = df.drop(columns=['pickup_datetime'])

total_amount_per_day = df_without_date.groupby(by='day').sum()[['fare_amount']]
total_amount_per_day = total_amount_per_day.reindex(index=day_order)

plt.figure(figsize=(10,5))
ax = sns.barplot(x=total_amount_per_day.index,y=total_amount_per_day['fare_amount'])
ax.set_xticklabels(day_order)
ax.set_ylabel("Pendapatan (USD)")
plt.title("Pendapatan perhari")

"""Pendapatan tinggi pada hari Kamis dan Jumat, sementara hari lainnya memiliki pendapatan yang lebih rendah. Selain itu, jumlah perjalanan tinggi pada hari Senin, yang cukup menarik.

### Revenue per month
"""

total_amount_per_month = df_without_date.groupby(by='month').sum()[['fare_amount']]
total_amount_per_month = total_amount_per_month.reindex(index=month_order)

plt.figure(figsize=(12,7))
ax = sns.barplot(x=total_amount_per_month.index,y=total_amount_per_month['fare_amount'])
ax.set_xticklabels(month_order)
ax.set_ylabel("Pendapatan (USD)")
plt.title("Pendapatan perbulan")

"""Pendapatan per bulan menunjukkan bahwa bulan-bulan musim panas, yaitu Juli, Agustus, dan September, memiliki pendapatan yang lebih rendah dibandingkan bulan-bulan lainnya, sebagaimana ditunjukkan oleh jumlah perjalanan per bulan.

## hypothesis test for fare amount relationship with passenger count

Seperti yang telah kita lihat sebelumnya, rata-rata tarif dengan jumlah penumpang menunjukkan nilai rata-rata yang sama untuk semua. Jadi, mari kita uji apakah ada hubungan di antara keduanya atau tidak.
"""

from scipy import stats

df.describe()[['fare_amount','passenger_count']]

df.groupby('passenger_count')[['fare_amount']].mean()

"""Seperti yang kita lihat, nilai jumlah penumpang dan tarif konsisten dan tidak menunjukkan adanya hubungan, tetapi hal ini bisa jadi karena pengambilan sampel secara acak. Untuk menentukan apakah nilai-nilai ini signifikan secara statistik, mari kita lakukan uji hipotesis.

**Hypothesis** <br>


**Null Hypothesis:** Tidak ada perbedaan antara rata-rata tarif berdasarkan jumlah penumpang. <br>
**Alternative Hypothesis:** Terdapat perbedaan antara rata-rata tarif berdasarkan jumlah penumpang.
"""

one_passenger = df[df['passenger_count'] == 1]['fare_amount']
two_passenger = df[df['passenger_count'] == 2]['fare_amount']
three_passenger = df[df['passenger_count'] == 3]['fare_amount']
four_passenger = df[df['passenger_count'] == 4]['fare_amount']
five_passenger = df[df['passenger_count'] == 5]['fare_amount']
six_passenger = df[df['passenger_count'] == 6]['fare_amount']

result = stats.f_oneway(one_passenger,two_passenger,three_passenger,four_passenger,five_passenger,six_passenger)
print("F-statistic:", result.statistic)
print("p-value:", result.pvalue)

"""Karena p-value lebih kecil dari tingkat signifikansi 0,05, maka kami menolak  null hypothesis dan menyimpulkan bahwa terdapat perbedaan signifikan dalam rata-rata tarif berdasarkan jumlah penumpang.

## Linear regression model

Setelah menghapus nilai yang hilang, selanjutnya analisis duplikat.
"""

df1 = df.copy()

df1.head()

df1.duplicated().sum()

"""Jadi, kami tidak memiliki duplikat atau nilai yang hilang. Sekarang, kami akan mencari dan menghapus outlier dari data karena model pembelajaran mesin terpengaruh oleh outlier tersebut."""

df1['pickup_datetime'] = pd.to_datetime(df1['pickup_datetime'],format='%m/%d/%Y %I:%M:%S %p')

df1['pickup_datetime'].head()

"""Sekarang fitur utama kami adalah jumlah tarif, jarak, dan jumlah penumpang, jadi mari kita lihat outlier di fitur-fitur tersebut.

### Removing Outline for LR model

Jadi, "fare_amount" dan 'distance' juga memiliki nilai negatif.
"""

sum(df1['distance_km']==0)

"""5 ribu perjalanan memiliki jarak 0."""

df1['fare_amount'].describe()

"""Nilai minimum adalah -52, jadi kita bisa mengubahnya menjadi 0, tetapi nilai maksimum adalah 499, yang tidak realistis."""

def outlier_imputer(df, column_list, iqr_factor):
    df_copy = df.copy()  # Work on a copy of the dataframe (Bekerja pada salinan dataframe)

    for col in column_list:
        q1 = df_copy[col].quantile(0.25)
        q3 = df_copy[col].quantile(0.75)
        iqr = q3 - q1
        upper_threshold = q3 + (iqr_factor * iqr)
        lower_threshold = q1 - (iqr_factor * iqr)

        print(col)
        print('q3:', q3)
        print('upper_threshold:', upper_threshold)

        print('q1:', q1)
        print('lower_threshold:', lower_threshold)

        # Filter out outliers
        df_copy = df_copy[(df_copy[col] <= upper_threshold) & (df_copy[col] >= lower_threshold)]
        print(df_copy[col].describe())
        print()

    return df_copy

df1 = outlier_imputer(df1, ['fare_amount','distance_km','passenger_count'], 1.5)

df1.shape

"""Sekarang kita memiliki kolom tanggal, bulan, dan hari.

Mari kita ubah menjadi huruf kecil dan temukan jam sibuk dalam data.
"""

df1.columns

df1['day'] = df1['day'].str.lower()
df1['month'] = df1['pickup_datetime'].dt.strftime('%b').str.lower()

df1.head()

"""### Feature Engineering: Creating Rush hour column"""

df1['rush_hour'] = df1['pickup_datetime'].dt.hour

df1.head()

df1.loc[df1['day'].isin(['sunday']), 'rush_hour'] = 1

def rush_hourizer(hour):
    if 6 <= hour['rush_hour'] < 10:
        val = 1
    elif 16 <= hour['rush_hour'] < 20:
        val = 1
    else:
        val = 0
    return val

df1.loc[(df1.day != 'sunday'), 'rush_hour'] = df1.apply(rush_hourizer, axis=1).astype('int32')

df1.head()

df1.columns

df2 = df1.drop(['Id', 'pickup_datetime', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','day', 'month'],axis=1)
df2.head()

df2['rush_hour'] = df2['rush_hour'].astype(float)

df2.shape

"""### Removing the rows with 0 distance"""

df2 = df2[df2['distance_km']!=0].reindex()

df2.shape

plt.figure(figsize=(6,4))
sns.heatmap(df2.corr(method='pearson'), annot=True, cmap='Reds')
plt.title('Correlation heatmap',
          fontsize=18)
plt.show()

"""### sklearn Imports"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics # For confusion matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

"""### Split between train and test sets"""

X = df2.drop(columns=['fare_amount'])

# Set y variable
y = df2[['fare_amount']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

"""### Scaling Data"""

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print('X_train scaled:', X_train_scaled)

"""### Fitting LR model"""

lr=LinearRegression()
lr.fit(X_train_scaled, y_train)

r_sq = lr.score(X_train_scaled, y_train)
print('Coefficient of determination:', r_sq)
y_pred_train = lr.predict(X_train_scaled)
print('R^2:', r2_score(y_train, y_pred_train))
print('MAE:', mean_absolute_error(y_train, y_pred_train))
print('MSE:', mean_squared_error(y_train, y_pred_train))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_pred_train)))

X_test_scaled = scaler.transform(X_test)

r_sq_test = lr.score(X_test_scaled, y_test)
print('Coefficient of determination:', r_sq_test)
y_pred_test = lr.predict(X_test_scaled)
print('R^2:', r2_score(y_test, y_pred_test))
print('MAE:', mean_absolute_error(y_test,y_pred_test))
print('MSE:', mean_squared_error(y_test, y_pred_test))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred_test)))

"""### Evaluating wether LR is good for this data or not?"""

results = pd.DataFrame(data={'actual': y_test['fare_amount'],
                             'predicted': y_pred_test.ravel()})
results['residual'] = results['actual'] - results['predicted']
results.head()

fig, ax = plt.subplots(figsize=(6, 6))
sns.set(style='whitegrid')
sns.scatterplot(x='actual',
                y='predicted',
                data=results,
                s=20,
                alpha=0.5,
                ax=ax
)
# Draw an x=y line to show what the results would be if the model were perfect
plt.plot([2.5,20], [2.5,20], c='red', linewidth=2)
plt.title('Aktual vs. Prediksi');

sns.scatterplot(x='predicted', y='residual', data=results)
plt.axhline(0, c='red')
plt.title('Scatterplot of residuals over predicted values')
plt.xlabel('predicted value')
plt.ylabel('residual value')
plt.show()

print(X_train['distance_km'].std())

# 2. Membagia koefisien model dengan standard deviation
print(2.959849 / X_train['distance_km'].std())

"""### Finding from the LR model

Jadi, menurut data, untuk setiap 1,42 km perjalanan, tarif meningkat rata-rata sebesar 2,95 dolar, atau untuk setiap 1 km perjalanan, tarif meningkat rata-rata sebesar 2,08 dolar.

## Training Model for prediction purpose
"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from xgboost import plot_importance

df1.head()

"""### Feature Engineering"""

# membuat kolom 'am_rush'
df1['am_rush'] = df1['pickup_datetime'].dt.hour

# membuat kolom 'daytime'
df1['daytime'] = df1['pickup_datetime'].dt.hour

# membuat kolom 'pm_rush'
df1['pm_rush'] = df1['pickup_datetime'].dt.hour

# membuat kolom 'nighttime'
df1['nighttime'] = df1['pickup_datetime'].dt.hour

def am_rush(hour):
    if 6 <= hour['am_rush'] < 10:
        val = 1
    else:
        val = 0
    return val

df1['am_rush'] = df1.apply(am_rush, axis=1)

def daytime(hour):
    if 10 <= hour['daytime'] < 16:
        val = 1
    else:
        val = 0
    return val

df1['daytime'] = df1.apply(daytime, axis=1)

def pm_rush(hour):
    if 16 <= hour['pm_rush'] < 20:
        val = 1
    else:
        val = 0
    return val

df1['pm_rush'] = df1.apply(pm_rush, axis=1)

def nighttime(hour):
    if 20 <= hour['nighttime'] < 24:
        val = 1
    elif 0 <= hour['nighttime'] < 6:
        val = 1
    else:
        val = 0
    return val

df1['nighttime'] = df1.apply(nighttime, axis=1)

df1.head()

drop_columns = ['Id','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','rush_hour']
df1 = df1.drop(drop_columns,axis=1)
df1.head()

df1 = pd.get_dummies(df1, drop_first=True)
df1.info()

X = df1.drop(['fare_amount'],axis=1)
y = df1[['fare_amount']]

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

X_train.shape

y_train.shape

"""## Random Forest Model"""

# Menerapkan Regresi Random Forest pada dataset
regressor = RandomForestRegressor(random_state=42)

# Sesuaikan regresor dengan data x dan y
regressor.fit(X_train, y_train)

cv_params = {'max_depth': [None],
             'max_features': [1.0],
             'max_samples': [0.7],
             'min_samples_leaf': [1],
             'min_samples_split': [2],
             'n_estimators': [300]
             }

# 3. Tentukan serangkaian metrik penilaian untuk menangkap
scoring = {'r2','accuracy'}

# 4. Inisialisasi objek GridSearchCV
rf1 = GridSearchCV(regressor, cv_params, scoring=scoring, cv=4,refit='r2')

# rf1.fit(X_train, y_train.ravel())

# rf1.best_score_

# rf1.best_params_

random_forest = RandomForestRegressor(max_depth= None,
 max_features= 1.0,
 max_samples= 0.7,
 min_samples_leaf= 1,
 min_samples_split= 2,
 n_estimators=300)

random_forest.fit(X_train,y_train)

# Evaluasi model
from sklearn.metrics import mean_squared_error, r2_score

# Membuat prediksi menggunakan data yang sudah ada atau data baru
predictions = random_forest.predict(X_test)

# Evaluasi data
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, predictions)
print(f'R-squared: {r2}')

"""### Random Forest Findings

Nilai R-squared adalah 0,58. Sekarang, mari kita coba XGBoost.

## Gradient Boost Model
"""

# 1. Inisialisasi XGBoost
xgb = XGBRegressor(objective ='reg:squarederror',random_state=42, learning_rate = 0.02, max_depth = 8,min_child_weight= 4,
 n_estimators = 200)

# 2. Buatlah dictionary hyperparameter yang akan disetel
cv_params = {'learning_rate': [0.1,0.01,0.02],
             'max_depth': [8,9,11],
             'min_child_weight': [2,3,4],
             'n_estimators': [500,200,300,600]
             }

# 3. Tentukan serangkaian metrik penilaian yang akan digunakan.
scoring = {'accuracy', 'r2'}

# 4. Inisialisasi objek GridSearchCV.
xgb1 = GridSearchCV(xgb, cv_params, scoring=scoring, cv=4, refit='r2')

y_train.shape

xgb.fit(X_train,y_train)

# Membuat prediksi pada data yang sama atau data baru
predictions = xgb.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, predictions)
print(f'R-squared: {r2}')

"""### XGB findings

XGB mencakupi 0.62 varian data atau dengan akurasi 62%, dimana hal ini merupakan hasil tertinggi

## Model Exporting
"""

X_test.head()

model.predict(X_test.head())

y_test.head(5)