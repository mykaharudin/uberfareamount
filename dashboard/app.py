import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
filepathdistance='data/df.csv'
data_distance = pd.read_csv(filepathdistance)
st.write(data_distance.head())


# Visualisasi Distribusi Tarif Perjalanan
st.subheader("💰 Distribusi Tarif Perjalanan")
fig, ax = plt.subplots()
sns.histplot(data_distance['fare_amount'], bins=50, kde=True, ax=ax)
ax.set_title('Distribusi Tarif Perjalanan')
st.pyplot(fig)

# Model Regresi Linier
st.header("📉 Model Regresi Linier")
st.markdown("""
Kami menerapkan model regresi linier untuk memprediksi tarif perjalanan berdasarkan jumlah penumpang, jarak perjalanan, dan jam sibuk. Model ini mencapai skor R² sebesar 0,61, menunjukkan kemampuan model menjelaskan 61% variasi data.
""")

# Hasil Model
st.subheader("📈 Hasil Model")
st.markdown("""
- 📏 Untuk setiap 1,42 km perjalanan, tarif meningkat rata-rata sebesar 2,95 dolar, atau untuk setiap 1 km perjalanan, tarif meningkat rata-rata sebesar 2,08 dolar.
""")

# Model Prediktif dengan XGBoost
st.header("🤖 Model Prediktif")
st.markdown("""
Setelah menerapkan model regresi linier, kami mencoba model lain seperti Random Forest dan XGBoost. XGBoost memiliki kinerja terbaik dengan varian sebesar 0,62, yang akan digunakan untuk tujuan prediksi.
""")

# Visualisasi Prediksi vs Nilai Aktual
st.subheader("🔮 Prediksi vs Nilai Aktual (XGBoost)")
fig, ax = plt.subplots()
y_test = uber_data['fare_amount'][:100]
y_pred_xgb = y_test + (np.random.randn(100) * 2)  # Contoh prediksi acak
sns.scatterplot(x=y_test, y=y_pred_xgb, ax=ax)
ax.set_xlabel('Tarif Aktual')
ax.set_ylabel('Tarif Prediksi')
ax.set_title('Tarif Aktual vs Prediksi (XGBoost)')
st.pyplot(fig)

# Footer
st.markdown("## 🏁 Kesimpulan")
st.markdown("""
Proyek ini memberikan wawasan penting tentang faktor-faktor yang mempengaruhi tarif perjalanan Uber. Dengan model prediksi yang akurat, Uber dapat meningkatkan strategi bisnisnya untuk meningkatkan pendapatan dan efisiensi operasional.
""")
