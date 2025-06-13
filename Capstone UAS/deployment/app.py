from flask import Flask, render_template, request, jsonify
import joblib, pandas as pd, os

app = Flask(__name__)

LABEL_MAP = {
    0: '0. Insufficient_Weight',
    1: '1. Normal_Weight',
    2: '2. Overweight_Level_I',
    3: '3. Overweight_Level_II',
    4: '4. Obesity_Type_I',
    5: '5. Obesity_Type_II',
    6: '6. Obesity_Type_III'
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models.pkl')
all_models = joblib.load(MODEL_PATH)

# Fitur dari form input (sebelum one-hot)
FORM_FEATURES = [
    'Age', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'SMOKE', 'CH2O',
    'family_history_with_overweight', 'FAF', 'TUE', 'CAEC',
    'Gender', 'MTRANS'
]

# Fitur yang digunakan untuk prediksi
USED_FEATURES = [
    'Age', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'SMOKE',
    'CH2O', 'family_history_with_overweight', 'FAF', 'TUE', 'CAEC',
    'Gender_Female', 'Gender_Male',
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking'
]

def convert_input(raw):
    # Numerik
    for k in ['Age', 'Height', 'Weight', 'CH2O', 'FAF', 'TUE']:
        raw[k] = float(raw[k]) if raw[k] else 0.0
    raw['NCP'] = int(raw['NCP']) if raw['NCP'] else 0

    # Binary (yes/no) ke angka
    for k in ['FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight']:
        val = str(raw.get(k, '')).lower()
        raw[k] = 1 if val == 'yes' else 0

    # Kategori ordinal (mapping ke int)
    CAEC_MAP = {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}
    CALC_MAP = {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}
    raw['CAEC'] = CAEC_MAP.get(str(raw['CAEC']).lower(), 0)
    raw['CALC'] = CALC_MAP.get(str(raw['CALC']).lower(), 0)

    # FCVC
    raw['FCVC'] = float(raw['FCVC']) if raw['FCVC'] else 1.0

    # Gender one-hot (boolean)
    raw['Gender_Female'] = 1 if raw['Gender'] == 'Female' else 0
    raw['Gender_Male']   = 1 if raw['Gender'] == 'Male' else 0

    # MTRANS one-hot (boolean)
    mtrans_options = [
        'Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'
    ]
    for opt in mtrans_options:
        col = f"MTRANS_{opt}"
        raw[col] = 1 if raw['MTRANS'] == opt else 0

    # Hapus field asli yang sudah di-one-hot
    raw.pop('Gender', None)
    raw.pop('MTRANS', None)

    # Pastikan semua fitur ada
    for col in USED_FEATURES:
        if col not in raw:
            raw[col] = 0

    return raw


@app.route('/')
def home():
    return render_template('index.html', label_map=LABEL_MAP)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        raw = {f: request.form.get(f) for f in FORM_FEATURES}
        raw = convert_input(raw)
        X = pd.DataFrame([raw], columns=USED_FEATURES)

        pred_dt = int(all_models['dt_model'].predict(X)[0])
        pred_rf = int(all_models['rf_model'].predict(X)[0])
        pred_knn = int(all_models['knn_model'].predict(X)[0])

        dt_label  = LABEL_MAP.get(pred_dt, f"Unknown ({pred_dt})")
        rf_label  = LABEL_MAP.get(pred_rf, f"Unknown ({pred_rf})")
        knn_label = LABEL_MAP.get(pred_knn, f"Unknown ({pred_knn})")

        return jsonify({
            'dt_label': dt_label,
            'rf_label': rf_label,
            'knn_label': knn_label
        })
    except Exception as e:
        app.logger.error("Predict error:", exc_info=e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
