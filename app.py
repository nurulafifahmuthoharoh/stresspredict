import streamlit as st
import joblib
import pandas as pd
from flask import Flask, jsonify, request
import threading

# Memuat model yang telah disimpan
model = joblib.load('naive_bayes_model.pkl')

# Fungsi untuk prediksi
def predict_stress(data):
    columns = [
        'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
        'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
        'noise_level', 'living_conditions', 'safety', 'basic_needs',
        'academic_performance', 'study_load', 'teacher_student_relationship',
        'future_career_concerns', 'social_support', 'peer_pressure',
        'extracurricular_activities', 'bullying'
    ]
    data_df = pd.DataFrame([data], columns=columns)
    
    # Melakukan prediksi berdasarkan input
    prediction = model.predict(data_df)
    proba = model.predict_proba(data_df)
    
    if prediction[0] == 0:
        return f"Tidak ada indikasi masalah kesehatan mental (probabilitas: {proba[0][0]*100:.2f}%)"
    elif prediction[0] == 1:
        return f"Stres sedang (probabilitas: {proba[0][1]*100:.2f}%)"
    elif prediction[0] == 2:
        return f"Stres tinggi (probabilitas: {proba[0][2]*100:.2f}%)"

# Inisialisasi Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data dari request JSON
        data = request.json
        # Mengonversi 'mental_health_history' dari string ke numeric
        data['mental_health_history'] = 1 if data['mental_health_history'] == "Yes" else 0

        # Membuat data input untuk prediksi
        input_data = [
            data['anxiety_level'],
            data['self_esteem'],
            data['mental_health_history'],
            data['depression'],
            data['headache'],
            data['blood_pressure'],
            data['sleep_quality'],
            data['breathing_problem'],
            data['noise_level'],
            data['living_conditions'],
            data['safety'],
            data['basic_needs'],
            data['academic_performance'],
            data['study_load'],
            data['teacher_student_relationship'],
            data['future_career_concerns'],
            data['social_support'],
            data['peer_pressure'],
            data['extracurricular_activities'],
            data['bullying']
        ]
        
        # Melakukan prediksi
        result = predict_stress(input_data)
        return jsonify({'result': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})


# Menjalankan Flask dalam thread terpisah agar bisa tetap menjalankan Streamlit
def run_flask():
    app.run(host='0.0.0.0', port=5000)

# Menjalankan Streamlit
def run_streamlit():
    st.title("Prediksi Level Stres")

    # Input fitur
    anxiety_level = st.slider("Level Kecemasan", 0, 30, 14)
    self_esteem = st.slider("Level Harga Diri", 0, 30, 20)
    mental_health_history = st.radio("Riwayat Kesehatan Mental", ("No", "Yes"))
    depression = st.slider("Level Depresi", 0, 30, 15)
    headache = st.slider("Level Sakit Kepala", 0, 5, 2)
    blood_pressure = st.slider("Tingkat Tekanan Darah", 0, 5, 3)
    sleep_quality = st.slider("Kualitas Tidur", 0, 5, 2)
    breathing_problem = st.slider("Masalah Pernapasan", 0, 5, 3)
    noise_level = st.slider("Tingkat Kebisingan", 0, 5, 3)
    living_conditions = st.slider("Kondisi Tempat Tinggal", 0, 5, 2)
    safety = st.slider("Keamanan", 0, 5, 3)
    basic_needs = st.slider("Kebutuhan Dasar", 0, 5, 3)
    academic_performance = st.slider("Performa Akademik", 0, 5, 3)
    study_load = st.slider("Beban Studi", 0, 5, 3)
    teacher_student_relationship = st.slider("Hubungan Guru-Murid", 0, 5, 3)
    future_career_concerns = st.slider("Kekhawatiran Karir Masa Depan", 0, 5, 2)
    social_support = st.slider("Dukungan Sosial", 0, 5, 3)
    peer_pressure = st.slider("Tekanan Teman Sebaya", 0, 5, 3)
    extracurricular_activities = st.slider("Kegiatan Ekstrakurikuler", 0, 5, 3)
    bullying = st.slider("Perundungan", 0, 5, 2)

    # Convert 'mental_health_history' from string to numeric
    mental_health_history = 1 if mental_health_history == "Yes" else 0

    # Membuat input data untuk prediksi
    input_data = [
        anxiety_level,
        self_esteem,
        mental_health_history,
        depression,
        headache,
        blood_pressure,
        sleep_quality,
        breathing_problem,
        noise_level,
        living_conditions,
        safety,
        basic_needs,
        academic_performance,
        study_load,
        teacher_student_relationship,
        future_career_concerns,
        social_support,
        peer_pressure,
        extracurricular_activities,
        bullying
    ]

    # Prediksi dengan model
    if st.button("Prediksi"):
        response = requests.post("http://localhost:5000/predict", json=input_data)
        result = response.json()
        st.write(result['result'])


if __name__ == '__main__':
    threading.Thread(target=run_flask).start()
    run_streamlit()
