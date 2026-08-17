from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("catboost_model.pkl")
metrics = joblib.load("model_metrics.pkl")

print("Fitur model:")
print(model.feature_names_)

print(type(model))
print(type(metrics))

# ============================================================
# HALAMAN UTAMA
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')


# ============================================================
# PREDIKSI
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    gender = request.form['gender']
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    bmi = float(request.form['bmi'])
    waist_size = float(request.form['waist_size'])
    blood_pressure = float(request.form['blood_pressure'])
    heart_rate = float(request.form['heart_rate'])
    cholesterol = float(request.form['cholesterol'])
    glucose = float(request.form['glucose'])
    insulin = float(request.form['insulin'])
    sleep_hours = float(request.form['sleep_hours'])
    sleep_quality = request.form['sleep_quality']
    work_hours = float(request.form['work_hours'])
    physical_activity = float(request.form['physical_activity'])
    daily_steps = float(request.form['daily_steps'])
    calorie_intake = float(request.form['calorie_intake'])
    sugar_intake = float(request.form['sugar_intake'])
    alcohol_consumption = request.form['alcohol_consumption']
    smoking_level = request.form['smoking_level']
    water_intake = float(request.form['water_intake'])
    screen_time = float(request.form['screen_time'])
    stress_level = int(request.form['stress_level'])
    diet_type = request.form['diet_type']
    exercise_type = request.form['exercise_type']
    device_usage = request.form['device_usage']
    meals_per_day = int(request.form['meals_per_day'])
    caffeine_intake = request.form['caffeine_intake']


    # ========================================================
    # MEMBUAT DATA INPUT
    # ========================================================

    data = pd.DataFrame([[
        age,
        gender,
        height,
        weight,
        bmi,
        waist_size,
        blood_pressure,
        heart_rate,
        cholesterol,
        glucose,
        insulin,
        sleep_hours,
        sleep_quality,
        work_hours,
        physical_activity,
        daily_steps,
        calorie_intake,
        sugar_intake,
        alcohol_consumption,
        smoking_level,
        water_intake,
        screen_time,
        stress_level,
        diet_type,
        exercise_type,
        device_usage,
        meals_per_day,
        caffeine_intake
    ]], columns=model.feature_names_)


    # ========================================================
    # PROBABILITAS
    # ========================================================

    prob = model.predict_proba(data)[0]

    prob_healthy = float(prob[0])
    prob_diseased = float(prob[1])


    # ========================================================
    # HASIL PREDIKSI
    # ========================================================

    # Threshold 0.30 sesuai kode yang kamu gunakan sekarang
    if prob_diseased >= 0.30:
        hasil = "Tidak Sehat"
    else:
        hasil = "Sehat"


    # ========================================================
    # KIRIM HASIL KE HALAMAN
    # ========================================================

    return render_template(
        'result.html',

        hasil=hasil,

        prob_healthy=prob_healthy * 100,
        prob_diseased=prob_diseased * 100,

        accuracy=metrics["accuracy"] * 100,

        precision_healthy=(
            metrics["precision_healthy"] * 100
        ),

        recall_healthy=(
            metrics["recall_healthy"] * 100
        ),

        f1_healthy=(
            metrics["f1_healthy"] * 100
        ),

        precision_diseased=(
            metrics["precision_diseased"] * 100
        ),

        recall_diseased=(
            metrics["recall_diseased"] * 100
        ),

        f1_diseased=(
            metrics["f1_diseased"] * 100
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)