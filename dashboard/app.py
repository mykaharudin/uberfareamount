import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Judul Aplikasi
st.title("🚕 Analisis Tarif Uber 🚕")

# Overview Proyek
st.header("📊 Project Overview")
st.markdown("""
Selamat datang di analisis tarif Uber! Proyek ini bertujuan untuk memahami dinamika tarif taksi Uber melalui eksplorasi data dan penerapan model linier, memberikan wawasan berharga bagi strategi bisnis Uber.
""")

# Deskripsi Dataset
st.header("📁 Dataset Overview")
st.markdown("""
Dataset ini terdiri dari 200.000 sampel yang mencakup 7 karakteristik utama. Sementara sebagian besar fitur berupa data numerik, 'pickup_datetime' dan 'id' berbentuk data integer.
""")

# Load data (misalnya dari file CSV yang sudah dibersihkan di Google Colab)
filepathdata='data/uber.csv'
uber_data = pd.read_csv(filepathdata)

# Contoh data
st.write("Berikut adalah contoh data yang dianalisis:")
st.write(uber_data.head())

# Temuan dari Analisis Data
st.header("🔍 Temuan dari Analisis Data")
st.markdown("""
Melalui berbagai visualisasi, kami menemukan beberapa pola menarik dalam data:
- 🚗 Mayoritas perjalanan memiliki jarak kurang dari 5 km.
- 💵 Tarif perjalanan sebagian besar berkisar antara 0 hingga 20 dolar.
- 🧑‍🤝‍🧑 Sebagian besar perjalanan memiliki penumpang antara 1 hingga 6 orang.
- ☀️ Bulan-bulan musim panas menunjukkan jumlah perjalanan yang lebih sedikit dibandingkan dengan bulan-bulan lainnya.
- 📅 Jumlah perjalanan harian tinggi pada hari Senin dan rendah pada akhir pekan.
- 💼 Pendapatan harian lebih rendah pada hari Senin dibandingkan hari Jumat.
- 📈 Tidak terdapat hubungan yang signifikan antara jumlah penumpang dan tarif perjalanan, namun uji hipotesis menunjukkan adanya korelasi.
""")

# Visualisasi Distribusi Jarak Perjalanan
st.subheader("📏 Distribusi Jarak Perjalanan")
filepathdistance='data/df1.csv'
data_distance = pd.read_csv(filepathdistance)
st.write(data_distance.head())


# Visualisasi Distribusi Tarif Perjalanan
st.subheader("💰 Distribusi Tarif Perjalanan")
fig, ax = plt.subplots()
sns.histplot(data_distance['fare_amount'], bins=50, kde=True, ax=ax)
ax.set_title('Distribusi Tarif Perjalanan')
st.image("https://github.com/mykaharudin/uberfareamount/blob/main/data/AktualvsPrediksi.png?raw=true")

# Model Regresi Linier
st.header("📉 Model Regresi Linier")
st.markdown("""
Kami menerapkan model regresi linier untuk memprediksi tarif perjalanan berdasarkan jumlah penumpang, jarak perjalanan, dan jam sibuk. Model ini mencapai skor R² sebesar 0,61, menunjukkan kemampuan model menjelaskan 61% variasi data.
""")
st.image("https://github.com/mykaharudin/uberfareamount/blob/main/data/Plotresidualprediksi.png?raw=true")

# Hasil Model
st.subheader("📈 Hasil Model")
st.image("https://github.com/mykaharudin/uberfareamount/blob/main/data/lr.png?raw=true")
st.markdown("""
- 📏 Untuk setiap 1,42 km perjalanan, tarif meningkat rata-rata sebesar 2,08 dolar, atau untuk setiap 1 km perjalanan, tarif meningkat rata-rata sebesar 2,08 dolar.
""")

# Model Prediktif dengan XGBoost
st.header("🤖 Model Pelatihan")

# Visualisasi Prediksi vs Nilai Aktual
st.subheader("🔮 Random Forest Model")
st.image("https://github.com/mykaharudin/uberfareamount/blob/main/data/Random%20Forest%20Model.PNG?raw=true")
st.markdown("Random Forest: Findings Nilai R-squared adalah 0,58.")


st.subheader("🔮 XGboost")
st.image("https://github.com/mykaharudin/uberfareamount/blob/main/data/XGboost.PNG?raw=true")
st.markdown("XGB findings: XGB mencakupi 0.62 varian data atau dengan akurasi 62%, dimana hal ini merupakan hasil tertinggi")

# Footer
st.markdown("## 🏁 Kesimpulan")
st.markdown("""
Proyek ini memberikan wawasan penting tentang faktor-faktor yang mempengaruhi tarif perjalanan Uber. Dengan model prediksi yang akurat, Uber dapat meningkatkan strategi bisnisnya untuk meningkatkan pendapatan dan efisiensi operasional.
""")


# Muat model
with open('data/gradient_boost.pickle', 'rb') as to_read:
    model = pickle.load(to_read)
    
# Daftar nama fitur yang digunakan saat melatih model
feature_names = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"]


# Fungsi untuk membuat prediksi
def predict(features):
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    return prediction[0]

# Aplikasi Streamlit
st.title("Prediksi Tarif Uber")

# Input dari pengguna
pickup_longitude = st.number_input("Pickup Longitude", value=40.7614327)
pickup_latitude = st.number_input("Pickup Latitude", value=-73.9798156)
dropoff_longitude = st.number_input("Dropoff Longitude", value=40.6513111)
dropoff_latitude = st.number_input("Dropoff Latitude", value=-73.8803331)
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)

# Tombol prediksi
if st.button("Prediksi"):
    features = {
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count
    }
    fare = predict(features)
    st.write(f"Tarif yang diprediksi adalah ${fare:.2f}")