import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
import time

# Load Dataset
df_clean = pd.read_csv('datasets/cleaned_dataframe.csv')
X = df_clean.drop('target', axis=1)
y = df_clean['target']

# Melakukan Oversampling dengan SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Normalisasi Data
sc = MinMaxScaler()
X_smote_normalized = sc.fit_transform(X_smote)

# Split Data
X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = train_test_split(X_smote_normalized, y_smote, test_size=0.2, random_state=42, stratify=y_smote)

# Load Model
model = pickle.load(open('models/rfc_normalization.pkl', 'rb'))

# Memprediksi Data
y_pred = model.predict(X_test_normalized)
akurasi = round(accuracy_score(y_test_normalized, y_pred)*100, 1)

st.set_page_config("Prediksi Penyakit Jantung", ":heart:")

st.title("Prediksi Penyakit Jantung")

st.subheader("Akurasi Model")
st.write(f"Dengan **Random Forest**: :green[**{akurasi}**] %")

st.sidebar.subheader("Masukkan Data")

age = st.sidebar.number_input(
    label="Usia",
    min_value=1,
    max_value=100,
    value=28,
    step=1
)

sb_sex = st.sidebar.selectbox(
    label="Jenis Kelamin (age)",
    options=["Pria", "Wanita"]
)

if sb_sex == "Pria":
    sex = 1
elif sb_sex == "Wanita":
    sex = 0
    
sb_cp = st.sidebar.selectbox(
    label="Tipe Nyeri Dada (cp)",
    options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)

if sb_cp == "Typical Angina":
    cp = 1
elif sb_cp == "Atypical Angina":
    cp = 2
elif sb_cp == "Non-anginal Pain":
    cp = 3
elif sb_cp == "Asymptomatic":
    cp = 4
    
trestbps = st.sidebar.number_input(
    label="Tekanan Darah mmHg (trestbps)",
    min_value=80,
    max_value=200
)

chol = st.sidebar.number_input(
    label="Kolesterol mg/dl (chol)",
    min_value=50,
    max_value=800,
)

sb_fbs = st.sidebar.selectbox(
    label="Apakah Gula Darah Puasa lebih dari 120 mg/dl? (fbs)",
    options=["Ya", "Tidak"]
)

if sb_fbs == "Ya":
    fbs = 1
elif sb_fbs == "Tidak":
    fbs = 0

sb_restecg = st.sidebar.selectbox(
    label="Hasil Elektrokardiografi (restecg)",
    options=["Normal", "ST-T wave abnormality", "Hypertrophy"]
)

if sb_restecg == "Normal":
    restecg = 0
elif sb_restecg == "ST-T wave abnormality":
    restecg = 1
elif sb_restecg == "Hypertrophy":
    restecg = 2
    
thalach = st.sidebar.number_input(
    label="Detak Jantung Maksimal (thalach)",
    min_value=80,
    max_value=200
)

sb_exang = st.sidebar.selectbox(
    label="Angina akibat Olahraga? (exang)",
    options=["Ya", "Tidak"]
)

if sb_exang == "Ya":
    exang = 1
elif sb_exang == "Tidak":
    exang = 0

oldpeak = round(st.sidebar.number_input(
    label="Depresi ST Induksi Olahraga (oldpeak)",
    min_value=0.0,
    max_value=10.0
), 2)

st.divider()
st.subheader("Hasil Masukkan Data")
input_col1, input_col2 = st.columns(2)
with input_col1:
    st.write(f"- Usia : **{age}**\n- Jenis Kelamin : **{sb_sex}**\n - Tipe Nyeri Dada : **{sb_cp}**\n - Tekanan Darah : **{trestbps}** mmHg\n - Kolesterol : **{chol}** mg/dl")
with input_col2:
    st.write(f"- Apakah Gula Darah Puasa > 120 mg/dl? : **{sb_fbs}**\n - Hasil Elektrokardiografi : **{sb_restecg}**\n - Detak Jantung Maksimal : **{thalach}**\n - Angina akibat Olahraga? : **{sb_exang}**\n - Depresi ST Induksi Olahraga : **{oldpeak}**")

predict_btn = st.button("Prediksi", type="primary")

if predict_btn:
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    input_data = sc.transform(input_data)
    prediction = model.predict(input_data) 
    
    # Menampilkan progress bar
    bar = st.progress(0)
    
    for i in range(101):
        bar.progress(i, f'Memproses... {i}%')
        time.sleep(0.005)
        if i == 100:
            bar.empty()
    
    st.divider()
    
    if prediction == 0:
        result = ":green[**Sehat**]"
    elif prediction == 1:
        result = ":orange[**Penyakit Jantung tingkat 1**]"
    elif prediction == 2:
        result = ":orange[**Penyakit Jantung tingkat 2**]"
    elif prediction == 3:
        result = ":red[**Penyakit Jantung tingkat 3**]"
    elif prediction == 4:
        result = ":red[**Penyakit Jantung tingkat 4**]"

    st.subheader("Hasil Prediksi: " + result)