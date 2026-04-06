from flask import Flask, render_template, request, jsonify
from flask import Flask, render_template, request, jsonify, redirect, url_for
import threading
import time
import pickle
import pandas as pd
from xgboost import XGBClassifier
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import os
from webdriver_manager.chrome import ChromeDriverManager

app = Flask(__name__)

# Load model from JSON format
model = XGBClassifier()
model_path = os.path.join(os.path.dirname(__file__), 'mi_model.json')
model.load_model(model_path)

preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessors.pkl')
with open(preprocessor_path, 'rb') as f:
    preprocessors = pickle.load(f)

label_encoder = preprocessors['label_encoder']
imputer = preprocessors['imputer']
feature_columns = preprocessors['feature_columns']

driver = None
wait = None
live_vitals = {}
live_lock = threading.Lock()
monitoring_active = False

def setup_driver(session_id):
    global driver, wait
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # Uncomment to run Chrome headless
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 30)
    driver.get('https://resusmonitor.com/set_id')
    session_input = wait.until(EC.presence_of_element_located((By.ID, 'session_id')))
    session_input.clear()
    session_input.send_keys(session_id)
    session_input.send_keys(Keys.RETURN)
    display_monitor_link = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn.btn-warning.btn-lg.col-12.stretched-link"))
    )
    display_monitor_link.click()
    time.sleep(5)

def get_live_vitals():
    global live_vitals, wait
    try:
        heart_rate = wait.until(EC.presence_of_element_located((By.ID, 'vs_hr'))).text.strip()
        spo2 = wait.until(EC.presence_of_element_located((By.ID, 'vs_spo2'))).text.strip()
        bp = wait.until(EC.presence_of_element_located((By.ID, 'vs_bp'))).text.strip()
        hr = float(heart_rate) if heart_rate.replace('.', '', 1).isdigit() else None
        sp = float(spo2) if spo2.replace('.', '', 1).isdigit() else None

        if bp and '/' in bp:
            parts = bp.split('/')
            try:
                sbp = float(parts[0].strip())
                dbp = float(parts[1].strip())
            except ValueError:
                sbp = None
                dbp = None
        else:
            sbp = None
            dbp = None

        with live_lock:
            live_vitals = {
                'Heart Rate': hr,
                'SpO2': sp,
                'Systolic_BP': sbp,
                'Diastolic_BP': dbp
            }
    except Exception:
        with live_lock:
            live_vitals = {
                'Heart Rate': None,
                'SpO2': None,
                'Systolic_BP': None,
                'Diastolic_BP': None
            }

def classify_vitals(vitals):
    risk_levels = []

    hr = vitals.get('Heart Rate')
    if hr is not None:
        if 60 <= hr <= 100:
            risk_levels.append("LOW")
        elif 50 <= hr < 60 or 101 <= hr <= 120:
            risk_levels.append("MEDIUM")
        else:
            risk_levels.append("HIGH")

    spo2 = vitals.get('SpO2')
    if spo2 is not None:
        if spo2 >= 95:
            risk_levels.append("LOW")
        elif 90 <= spo2 <= 94:
            risk_levels.append("MEDIUM")
        else:
            risk_levels.append("HIGH")

    sbp = vitals.get('Systolic_BP')
    dbp = vitals.get('Diastolic_BP')
    if sbp is not None and dbp is not None:
        if 100 <= sbp <= 129 and 60 <= dbp <= 80:
            risk_levels.append("LOW")
        elif (130 <= sbp <= 159) or (81 <= dbp <= 99) or (90 <= sbp < 100):
            risk_levels.append("MEDIUM")
        else:
            risk_levels.append("HIGH")

    if "HIGH" in risk_levels:
        return "HIGH"
    elif "MEDIUM" in risk_levels:
        return "MEDIUM"
    return "LOW"

def prediction_thread(static_data):
    global monitoring_active
    while monitoring_active:
        get_live_vitals()
        time.sleep(3)

@app.route('/', methods=['GET', 'POST'])
def index():
    global monitoring_active
    if request.method == 'POST':
        try:
            static_data = {
                'age': float(request.form['age']),
                'gender': request.form['gender'],
                'resp_rate': float(request.form['resp_rate']),
                'temperature': float(request.form['temperature']),
                'troponin': float(request.form['troponin']),
                'ck_mb': float(request.form['ck_mb']),
                'ldh': float(request.form['ldh']),
                'cholesterol': float(request.form['cholesterol']),
                'hdl': float(request.form['hdl']),
                'ldl': float(request.form['ldl']),
                'triglycerides': float(request.form['triglycerides']),
                'ecg_st_elevation': int(request.form['ecg_st_elevation']),
                'ecg_q_waves': int(request.form['ecg_q_waves']),
                'family_history_mi': int(request.form['family_history_mi']),
                'diabetes': int(request.form['diabetes']),
                'hypertension': int(request.form['hypertension']),
                'smoker': int(request.form['smoker']),
                'obesity': int(request.form['obesity']),
                'physical_inactivity': int(request.form['physical_inactivity']),
                'alcohol_consumption': int(request.form['alcohol_consumption']),
                'stress_level': int(request.form['stress_level']),
                'session_id': request.form['session_id']
            }
            static_data['gender_encoded'] = label_encoder.transform([static_data['gender']])[0]

            if not monitoring_active:
                setup_driver(static_data['session_id'])
                monitoring_active = True
                t = threading.Thread(target=prediction_thread, args=(static_data,), daemon=True)
                t.start()

            return render_template('live_monitor.html', static_data=static_data)
        except Exception as ex:
            return f"Error: {ex}"

    return render_template('index.html')

@app.route('/get_risk')
def get_risk():
    with live_lock:
        current_vitals = live_vitals.copy()

    if None in current_vitals.values():
        return jsonify({'status': 'incomplete'})

    static_data = request.args
    try:
        age = float(static_data.get('age'))
        gender_encoded = int(static_data.get('gender_encoded'))
        resp_rate = float(static_data.get('resp_rate'))
        temperature = float(static_data.get('temperature'))
        troponin = float(static_data.get('troponin'))
        ck_mb = float(static_data.get('ck_mb'))
        ldh = float(static_data.get('ldh'))
        cholesterol = float(static_data.get('cholesterol'))
        hdl = float(static_data.get('hdl'))
        ldl = float(static_data.get('ldl'))
        triglycerides = float(static_data.get('triglycerides'))
        ecg_st_elevation = int(static_data.get('ecg_st_elevation'))
        ecg_q_waves = int(static_data.get('ecg_q_waves'))
        family_history_mi = int(static_data.get('family_history_mi'))
        diabetes = int(static_data.get('diabetes'))
        hypertension = int(static_data.get('hypertension'))
        smoker = int(static_data.get('smoker'))
        obesity = int(static_data.get('obesity'))
        physical_inactivity = int(static_data.get('physical_inactivity'))
        alcohol_consumption = int(static_data.get('alcohol_consumption'))
        stress_level = int(static_data.get('stress_level'))

        input_df = pd.DataFrame([[
            age,
            gender_encoded,
            current_vitals['Heart Rate'],
            current_vitals['Systolic_BP'],
            current_vitals['Diastolic_BP'],
            current_vitals['SpO2'],
            resp_rate,
            temperature,
            troponin,
            ck_mb,
            ldh,
            cholesterol,
            hdl,
            ldl,
            triglycerides,
            ecg_st_elevation,
            ecg_q_waves,
            family_history_mi,
            diabetes,
            hypertension,
            smoker,
            obesity,
            physical_inactivity,
            alcohol_consumption,
            stress_level
        ]], columns=feature_columns)

        input_imputed = imputer.transform(input_df)
        pred_proba = model.predict_proba(input_imputed)[0][1]

        low_threshold = 0.15
        high_threshold = 0.5
        if pred_proba < low_threshold:
            ml_risk = "LOW"
        elif pred_proba < high_threshold:
            ml_risk = "MEDIUM"
        else:
            ml_risk = "HIGH"

        vitals_risk = classify_vitals(current_vitals)
        final_risk = max([ml_risk, vitals_risk], key=lambda x: ["LOW", "MEDIUM", "HIGH"].index(x))

        return jsonify({
            'status': 'complete',
            'final_risk': final_risk,
            'ml_risk': ml_risk,
            'vitals_risk': vitals_risk,
            'probability': f"{pred_proba:.2%}",
            'vitals': current_vitals
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_monitoring')
def stop_monitoring():
    global driver, monitoring_active
    monitoring_active = False
    if driver:
        driver.quit()
        driver = None
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
