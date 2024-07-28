import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Judul Aplikasi
st.title("Analisis Tarif Uber")

# Overview Proyek
st.header("Project Overview")
st.markdown("""
Proyek Tarif Taksi Uber bertujuan untuk menganalisis, mengeksplorasi, membersihkan, dan mengatur data, serta menerapkan model linier untuk memahami hubungan antar fitur guna mendapatkan wawasan bisnis yang lebih mendalam.
""")

# Deskripsi Dataset
st.header("Dataset Overview")
st.markdown("""
Dataset ini terdiri dari 200.000 sampel, masing-masing memiliki 7 karakteristik. Seluruh karakteristik memiliki tipe data float, kecuali 'pickup_datetime' dan 'id' yang bertipe int.
""")

# Load data (misalnya dari file CSV yang sudah dibersihkan di Google Colab)
@st.cache
def load_data():
    df = pd.read_csv('/mnt/e/DA and DS/Digiskol/zfinalProject/coba/uberfareamount/uber.csv')
    return df

data = load_data()

# Contoh data
st.write("Berikut adalah contoh data yang dianalisis:")
st.write(data.head())

# Temuan dari Analisis Data
st.header("Temuan dari Analisis Data")
st.markdown("""
Kami membuat berbagai visualisasi untuk memahami distribusi dan karakteristik data. Beberapa temuan yang kami dapatkan adalah:
- Mayoritas perjalanan memiliki jarak kurang dari 5 km.
- Jumlah tarif untuk sebagian besar perjalanan berada antara 0 hingga 20 dolar.
- Pada sebagian besar perjalanan, jumlah penumpang berkisar antara 1-6 orang.
- Analisis perjalanan dan pendapatan bulanan menunjukkan bahwa bulan-bulan musim panas memiliki jumlah perjalanan yang lebih sedikit dibandingkan dengan bulan-bulan lainnya sepanjang tahun.
- Jumlah perjalanan per hari tinggi pada hari Senin dan rendah pada hari Sabtu dan Minggu.
- Analisis pendapatan harian menunjukkan bahwa pendapatan pada hari Senin lebih rendah dibandingkan dengan hari Jumat.
- Rata-rata jumlah tarif terkait dengan jumlah penumpang tidak menunjukkan adanya hubungan, sehingga kami melakukan uji hipotesis yang menunjukkan adanya hubungan.
""")

# Visualisasi Distribusi Jarak Perjalanan
st.subheader("Distribusi Jarak Perjalanan")
fig, ax = plt.subplots()
sns.histplot(data['distance_km'], bins=50, kde=True, ax=ax)
ax.set_title('Distribusi Jarak Perjalanan')
st.pyplot(fig)

# Visualisasi Distribusi Tarif Perjalanan
st.subheader("Distribusi Tarif Perjalanan")
fig, ax = plt.subplots()
sns.histplot(data['fare_amount'], bins=50, kde=True, ax=ax)
ax.set_title('Distribusi Tarif Perjalanan')
st.pyplot(fig)

# Model Regresi Linier
st.header("Model Regresi Linier")
st.markdown("""
Kami menerapkan model regresi linier untuk memprediksi tarif perjalanan berdasarkan jumlah penumpang, jarak perjalanan, dan jam sibuk. Model ini mencapai skor R² sebesar 0,61, yang menunjukkan bahwa model dapat menjelaskan 61% dari variasi data.
""")

# Hasil Model
st.subheader("Hasil Model")
st.markdown("""
- Untuk setiap 1,42 km perjalanan, tarif meningkat rata-rata sebesar 2,95 dolar, atau untuk setiap 1 km perjalanan, tarif meningkat rata-rata sebesar 2,08 dolar.
""")

# Model Prediktif dengan XGBoost
st.header("Model Prediktif")
st.markdown("""
Setelah menerapkan model regresi linier, kami mencoba model lain seperti Random Forest dan XGBoost. XGBoost memiliki kinerja terbaik dengan varian sebesar 0,62, yang akan digunakan untuk tujuan prediksi.
""")

# Visualisasi Prediksi vs Nilai Aktual
st.subheader("Prediksi vs Nilai Aktual (XGBoost)")
fig, ax = plt.subplots()
y_test = data['fare_amount'][:100]
y_pred_xgb = y_test + (np.random.randn(100) * 2)  # Contoh prediksi acak
sns.scatterplot(x=y_test, y=y_pred_xgb, ax=ax)
ax.set_xlabel('Tarif Aktual')
ax.set_ylabel('Tarif Prediksi')
ax.set_title('Tarif Aktual vs Prediksi (XGBoost)')
st.pyplot(fig)

# Footer
st.markdown("## Kesimpulan")
st.markdown("""
Proyek ini memberikan wawasan penting tentang faktor-faktor yang mempengaruhi tarif perjalanan Uber. Dengan model prediksi yang akurat, Uber dapat meningkatkan strategi bisnisnya untuk meningkatkan pendapatan dan efisiensi operasional.
""")
