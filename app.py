from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3
import datetime
import json
from flask import session

app = Flask(__name__)
app.secret_key = 'healthai_secret_key_2025'
DB_PATH = 'healthai.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, age INTEGER, gender TEXT,
        phone TEXT, email TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER, disease TEXT NOT NULL,
        risk_level TEXT NOT NULL, probability REAL NOT NULL,
        input_data TEXT NOT NULL, top_factors TEXT,
        predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients(id))''')
    conn.commit()
    conn.close()

init_db()

# Load both models
with open('model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

with open('heart_model.pkl', 'rb') as f:
    heart_model = pickle.load(f)

DIABETES_FEATURES = [
    'Pregnancies', 'Glucose', 'Blood Pressure',
    'Skin Thickness', 'Insulin', 'BMI',
    'Diabetes Pedigree', 'Age'
]

HEART_FEATURES = [
    'Age', 'Sex', 'Chest Pain', 'Resting BP',
    'Cholesterol', 'Fasting BS', 'Rest ECG',
    'Max Heart Rate', 'Exercise Angina',
    'ST Depression', 'Slope', 'Vessels', 'Thal'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        values = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age']),
        ]
        features = np.array([values])
        prediction   = diabetes_model.predict(features)[0]
        probability  = diabetes_model.predict_proba(features)[0][prediction] * 100
        result       = "High Risk" if prediction == 1 else "Low Risk"
        color        = "danger"    if prediction == 1 else "success"

        importances  = diabetes_model.feature_importances_
        feature_list = sorted([
            {'name': n, 'importance': round((i / sum(importances)) * 100, 1)}
            for n, i in zip(DIABETES_FEATURES, importances)
        ], key=lambda x: x['importance'], reverse=True)

        inputs = {
            'age': values[7], 'bmi': values[5],
            'glucose': values[1], 'blood_pressure': values[2]
        }

        # ── fetch previous prediction for this disease ──
        patient_id   = request.form.get('patient_id') or request.args.get('patient_id')
        prev_prediction = None
        if patient_id:
            conn = get_db()
            prev = conn.execute(
                "SELECT * FROM predictions WHERE patient_id=? AND disease='Diabetes' ORDER BY predicted_at DESC LIMIT 1",
                (patient_id,)
            ).fetchone()
            conn.close()
            if prev:
                prev_prediction = dict(prev)

        # ── get all patients for Save dropdown ──
        conn = get_db()
        all_patients = conn.execute('SELECT id, name, age FROM patients ORDER BY name').fetchall()
        conn.close()

        return render_template('result.html',
                               result=result,
                               disease="Diabetes",
                               probability=round(probability, 2),
                               color=color,
                               features=feature_list,
                               inputs=inputs,
                               prev_prediction=prev_prediction,
                               patient_id=int(patient_id) if patient_id else None,
                               all_patients=all_patients)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        values = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal']),
        ]
        features     = np.array([values])
        prediction   = heart_model.predict(features)[0]
        probability  = heart_model.predict_proba(features)[0][prediction] * 100
        result       = "High Risk" if prediction == 1 else "Low Risk"
        color        = "danger"    if prediction == 1 else "success"

        importances  = heart_model.feature_importances_
        feature_list = sorted([
            {'name': n, 'importance': round((i / sum(importances)) * 100, 1)}
            for n, i in zip(HEART_FEATURES, importances)
        ], key=lambda x: x['importance'], reverse=True)

        inputs = {
            'age': values[0], 'bmi': 'N/A',
            'glucose': 'N/A', 'blood_pressure': values[3]
        }

        patient_id = request.form.get('patient_id') or request.args.get('patient_id')
        prev_prediction = None
        if patient_id:
            conn = get_db()
            prev = conn.execute(
                "SELECT * FROM predictions WHERE patient_id=? AND disease='Heart Disease' ORDER BY predicted_at DESC LIMIT 1",
                (patient_id,)
            ).fetchone()
            conn.close()
            if prev:
                prev_prediction = dict(prev)

        conn = get_db()
        all_patients = conn.execute('SELECT id, name, age FROM patients ORDER BY name').fetchall()
        conn.close()

        return render_template('result.html',
                               result=result,
                               disease="Heart Disease",
                               probability=round(probability, 2),
                               color=color,
                               features=feature_list,
                               inputs=inputs,
                               prev_prediction=prev_prediction,
                               patient_id=int(patient_id) if patient_id else None,
                               all_patients=all_patients)
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/chat')
def chat():
    return render_template('chat.html')
@app.route('/emergency')
def emergency():
    return render_template('emergency.html')
@app.route('/signlanguage')
def signlanguage():
    return render_template('signlanguage.html')
@app.route('/hospital')
def hospital():
    return render_template('hospital.html')
@app.route('/voice')
def voice():
    return render_template('voice.html')
@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from flask import send_file
import io
import datetime

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/report')
def report():
    return render_template('report.html')
@app.route('/vitals')
def vitals():
    return render_template('vitals.html')

@app.route('/generate_report', methods=['POST'])
def generate_report():
    name = request.form.get('name', 'Patient')
    age  = request.form.get('age', 'N/A')
    disease_results = [
        ('Diabetes',      request.form.get('diabetes_risk', '72'), 'High Risk',  '#FF4444'),
        ('Heart Disease', request.form.get('heart_risk',    '34'), 'Low Risk',   '#00AA44'),
        ('Liver Disease', request.form.get('liver_risk',    '58'), 'Moderate',   '#FF8800'),
        ('Kidney Disease',request.form.get('kidney_risk',   '21'), 'Low Risk',   '#00AA44'),
    ]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story  = []

    # Title
    title_style = styles['Title']
    story.append(Paragraph("AI Health Prediction Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))

    # Patient info
    story.append(Paragraph("Patient Information", styles['Heading2']))
    patient_data = [
        ['Name', name],
        ['Age',  age],
        ['Report Date', datetime.date.today().strftime('%d-%m-%Y')],
        ['Report Type', 'AI Early Disease Risk Prediction'],
    ]
    pt = Table(patient_data, colWidths=[2*inch, 4*inch])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), colors.HexColor('#f8f9fa')),
        ('FONTNAME',   (0,0),(0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 11),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.white, colors.HexColor('#f0f4f8')]),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('PADDING',    (0,0),(-1,-1), 8),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.3*inch))

    # Risk results
    story.append(Paragraph("Disease Risk Prediction Results", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))

    risk_data = [['Disease', 'Risk Score', 'Status', 'Action Required']]
    actions = {
        'High Risk':  'Consult specialist immediately',
        'Moderate':   'Schedule checkup within 2 weeks',
        'Low Risk':   'Maintain healthy lifestyle',
    }
    for disease, score, status, _ in disease_results:
        risk_data.append([disease, score+'%', status, actions.get(status, 'Consult doctor')])

    rt = Table(risk_data, colWidths=[1.6*inch, 1.2*inch, 1.2*inch, 3*inch])
    rt.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
        ('PADDING',     (0,0), (-1,-1), 8),
        ('ALIGN',       (1,1), (1,-1),  'CENTER'),
    ]))
    story.append(rt)
    story.append(Spacer(1, 0.3*inch))

    # Recommendations
    story.append(Paragraph("General Health Recommendations", styles['Heading2']))
    recs = [
        "1. Maintain a balanced diet rich in vegetables, whole grains, and lean proteins.",
        "2. Exercise for at least 30 minutes daily — walking, swimming or cycling.",
        "3. Monitor blood glucose and blood pressure regularly.",
        "4. Avoid smoking and limit alcohol consumption.",
        "5. Get 7-8 hours of quality sleep every night.",
        "6. Schedule annual comprehensive health checkups.",
        "7. Stay hydrated — drink at least 8 glasses of water daily.",
    ]
    for r in recs:
        story.append(Paragraph(r, styles['Normal']))
        story.append(Spacer(1, 0.05*inch))

    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        "⚠️ Disclaimer: This report is generated by an AI system for educational purposes only. "
        "It is NOT a substitute for professional medical diagnosis. Please consult a qualified "
        "healthcare professional for proper medical advice.",
        styles['Normal']
    ))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f'health_report_{name.replace(" ","_")}.pdf',
                     mimetype='application/pdf')


@app.route('/identify', methods=['POST'])
def identify():
    selected = request.json.get('symptoms', [])

    diseases = {
        'Diabetes': {
            'symptoms': ['frequent urination', 'excessive thirst', 'fatigue', 'blurred vision', 'slow healing', 'weight loss'],
            'color': 'blue', 'emoji': '🩸'
        },
        'Heart Disease': {
            'symptoms': ['chest pain', 'shortness of breath', 'fatigue', 'dizziness', 'irregular heartbeat', 'swollen legs'],
            'color': 'red', 'emoji': '❤️'
        },
        'Hypertension': {
            'symptoms': ['headache', 'dizziness', 'blurred vision', 'chest pain', 'shortness of breath', 'fatigue'],
            'color': 'orange', 'emoji': '⚡'
        },
        'Anemia': {
            'symptoms': ['fatigue', 'weakness', 'pale skin', 'shortness of breath', 'dizziness', 'headache'],
            'color': 'purple', 'emoji': '🫀'
        }
    }

    results = []
    for disease, info in diseases.items():
        matches = [s for s in selected if s in info['symptoms']]
        if matches:
            score = round(len(matches) / len(info['symptoms']) * 100)
            results.append({
                'disease': disease,
                'score': score,
                'matches': len(matches),
                'total': len(info['symptoms']),
                'emoji': info['emoji']
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return json.dumps({'results': results})   
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message', '').lower()

    responses = {
        'diabetes': "Diabetes risk increases with high glucose, obesity, and family history. Key indicators: fasting glucose >126 mg/dL, BMI >30, age >45.",
        'heart': "Heart disease risk factors include high cholesterol, high blood pressure, smoking, and sedentary lifestyle. Keep BP below 120/80.",
        'symptoms': "Common early warning signs: frequent urination, fatigue, chest pain, shortness of breath, blurred vision. Always consult a doctor.",
        'diet': "For disease prevention: eat more vegetables, whole grains, and lean protein. Reduce sugar, salt, and processed foods.",
        'exercise': "Aim for 150 minutes of moderate exercise weekly. Walking, swimming, and cycling are excellent for heart and diabetes prevention.",
        'bmi': "BMI 18.5-24.9 = Normal. 25-29.9 = Overweight. 30+ = Obese. High BMI increases diabetes and heart disease risk.",
        'blood pressure': "Normal BP is below 120/80 mmHg. High BP (>140/90) significantly increases heart attack and stroke risk.",
        'cholesterol': "Total cholesterol should be below 200 mg/dL. LDL (bad) below 100, HDL (good) above 60.",
        'glucose': "Normal fasting glucose: 70-100 mg/dL. Pre-diabetes: 100-125. Diabetes: 126+.",
        'emergency': "If you experience chest pain, difficulty breathing, or sudden weakness — call emergency services (112/911) immediately!",
        'hello': "Hello! I'm your AI Health Assistant. Ask me about diabetes, heart disease, symptoms, diet, or exercise!",
        'hi': "Hi there! I'm here to help with health questions. What would you like to know?",
    }

    reply = "I'm not sure about that. Try asking about: diabetes, heart disease, symptoms, diet, exercise, BMI, or blood pressure."

    for keyword, response in responses.items():
        if keyword in user_message:
            reply = response
            break

    return json.dumps({'reply': reply})
# Load liver and kidney models
with open('liver_model.pkl', 'rb') as f:
    liver_model = pickle.load(f)

with open('kidney_model.pkl', 'rb') as f:
    kidney_model = pickle.load(f)

LIVER_FEATURES = ['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin',
                  'Alkaline Phosphotase', 'Alamine Aminotransferase',
                  'Aspartate Aminotransferase', 'Total Proteins',
                  'Albumin', 'Albumin and Globulin Ratio']

KIDNEY_FEATURES = ['Age', 'Blood Pressure', 'Specific Gravity', 'Albumin',
                   'Sugar', 'Red Blood Cells', 'Pus Cell', 'Pus Cell Clumps',
                   'Bacteria', 'Blood Glucose Random', 'Blood Urea',
                   'Serum Creatinine', 'Sodium', 'Potassium',
                   'Haemoglobin', 'Packed Cell Volume',
                   'White Blood Cell Count', 'Red Blood Cell Count',
                   'Hypertension', 'Diabetes Mellitus',
                   'Coronary Artery Disease', 'Appetite',
                   'Pedal Edema', 'Anemia']

@app.route('/liver')
def liver():
    return render_template('liver.html')


@app.route('/predict/liver', methods=['POST'])
def predict_liver():
    try:
        values = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['total_bilirubin']),
            float(request.form['direct_bilirubin']),
            float(request.form['alkaline']),
            float(request.form['alamine']),
            float(request.form['aspartate']),
            float(request.form['total_proteins']),
            float(request.form['albumin']),
            float(request.form['ag_ratio']),
        ]
        features     = np.array([values])
        prediction   = liver_model.predict(features)[0]
        probability  = liver_model.predict_proba(features)[0][1] * 100
        result       = "High Risk" if prediction == 1 else "Low Risk"
        color        = "danger"    if prediction == 1 else "success"

        importances  = liver_model.feature_importances_
        feature_list = sorted([
            {'name': n, 'importance': round((i / sum(importances)) * 100, 1)}
            for n, i in zip(LIVER_FEATURES, importances)
        ], key=lambda x: x['importance'], reverse=True)

        inputs = {
            'age': values[0], 'bmi': 'N/A',
            'glucose': values[9], 'blood_pressure': 'N/A'
        }

        patient_id = request.form.get('patient_id') or request.args.get('patient_id')
        prev_prediction = None
        if patient_id:
            conn = get_db()
            prev = conn.execute(
                "SELECT * FROM predictions WHERE patient_id=? AND disease='Liver Disease' ORDER BY predicted_at DESC LIMIT 1",
                (patient_id,)
            ).fetchone()
            conn.close()
            if prev:
                prev_prediction = dict(prev)

        conn = get_db()
        all_patients = conn.execute('SELECT id, name, age FROM patients ORDER BY name').fetchall()
        conn.close()

        return render_template('result.html',
                               result=result,
                               disease="Liver Disease",
                               probability=round(probability, 2),
                               color=color,
                               features=feature_list,
                               inputs=inputs,
                               prev_prediction=prev_prediction,
                               patient_id=int(patient_id) if patient_id else None,
                               all_patients=all_patients)
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

KIDNEY_FEATURES = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
                   'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                   'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

KIDNEY_DISPLAY = ['Age', 'Blood Pressure', 'Specific Gravity', 'Albumin',
                  'Sugar', 'Red Blood Cells', 'Pus Cell', 'Pus Cell Clumps',
                  'Bacteria', 'Blood Glucose', 'Blood Urea', 'Serum Creatinine',
                  'Sodium', 'Potassium', 'Haemoglobin', 'Packed Cell Volume',
                  'WBC Count', 'RBC Count', 'Hypertension', 'Diabetes',
                  'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia']

@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    try:
        values = [
            float(request.form['age']),
            float(request.form['bp']),
            float(request.form['sg']),
            float(request.form['al']),
            float(request.form['su']),
            float(request.form['rbc']),
            float(request.form['pc']),
            float(request.form['pcc']),
            float(request.form['ba']),
            float(request.form['bgr']),
            float(request.form['bu']),
            float(request.form['sc']),
            float(request.form['sod']),
            float(request.form['pot']),
            float(request.form['hemo']),
            float(request.form['pcv']),
            float(request.form['wc']),
            float(request.form['rc']),
            float(request.form['htn']),
            float(request.form['dm']),
            float(request.form['cad']),
            float(request.form['appet']),
            float(request.form['pe']),
            float(request.form['ane']),
        ]
        features     = np.array([values])
        prediction   = kidney_model.predict(features)[0]
        probability  = kidney_model.predict_proba(features)[0][1] * 100
        result       = "High Risk" if prediction == 1 else "Low Risk"
        color        = "danger"    if prediction == 1 else "success"

        importances  = kidney_model.feature_importances_
        feature_list = sorted([
            {'name': n, 'importance': round((i / sum(importances)) * 100, 1)}
            for n, i in zip(KIDNEY_DISPLAY, importances)
        ], key=lambda x: x['importance'], reverse=True)

        inputs = {
            'age': values[0], 'bmi': 'N/A',
            'glucose': values[9], 'blood_pressure': values[1]
        }

        patient_id = request.form.get('patient_id') or request.args.get('patient_id')
        prev_prediction = None
        if patient_id:
            conn = get_db()
            prev = conn.execute(
                "SELECT * FROM predictions WHERE patient_id=? AND disease='Kidney Disease' ORDER BY predicted_at DESC LIMIT 1",
                (patient_id,)
            ).fetchone()
            conn.close()
            if prev:
                prev_prediction = dict(prev)

        conn = get_db()
        all_patients = conn.execute('SELECT id, name, age FROM patients ORDER BY name').fetchall()
        conn.close()

        return render_template('result.html',
                               result=result,
                               disease="Kidney Disease",
                               probability=round(probability, 2),
                               color=color,
                               features=feature_list,
                               inputs=inputs,
                               prev_prediction=prev_prediction,
                               patient_id=int(patient_id) if patient_id else None,
                               all_patients=all_patients)
    except Exception as e:
        return f"Error: {str(e)}"

ROUTES_TO_ADD = '''

@app.route('/set_language/<lang>')
def set_language(lang):
    ALLOWED = ['en','ta','hi','te','kn','ml','fr','ar']
    session['lang'] = lang if lang in ALLOWED else 'en'
    return redirect(request.referrer or '/')

@app.route('/patients')
def patients():
    conn = get_db()
    rows = conn.execute('SELECT * FROM patients ORDER BY created_at DESC').fetchall()
    patients_list = []
    for p in rows:
        latest = conn.execute(
            "SELECT * FROM predictions WHERE patient_id=? ORDER BY predicted_at DESC LIMIT 1",
            (p['id'],)).fetchone()
        d = dict(p)
        d['latest_risk']    = latest['risk_level']    if latest else None
        d['latest_disease'] = latest['disease']       if latest else None
        patients_list.append(d)
    conn.close()
    return render_template('patients.html', patients=patients_list)

@app.route('/add_patient', methods=['POST'])
def add_patient():
    conn = get_db()
    conn.execute(
        'INSERT INTO patients (name, age, gender, phone, email) VALUES (?,?,?,?,?)',
        (request.form['name'], request.form.get('age',0),
         request.form.get('gender',''), request.form.get('phone',''),
         request.form.get('email','')))
    conn.commit()
    pid = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    conn.close()
    return redirect(f'/patient/{pid}')

@app.route('/patient/<int:pid>')
def patient_detail(pid):
    conn = get_db()
    patient = conn.execute('SELECT * FROM patients WHERE id=?', (pid,)).fetchone()
    if not patient:
        return "Patient not found", 404
    history = conn.execute(
        'SELECT * FROM predictions WHERE patient_id=? ORDER BY predicted_at DESC',
        (pid,)).fetchall()

    progress = {}
    disease_data = {}
    for h in reversed(history):
        d = h['disease']
        if d not in disease_data:
            disease_data[d] = []
        disease_data[d].append(h['probability'])

    for disease, probs in disease_data.items():
        if len(probs) >= 2:
            first  = round(probs[0],  1)
            latest = round(probs[-1], 1)
            diff   = round(abs(latest - first), 1)
            trend  = 'improved' if latest < first else ('worsened' if latest > first else 'stable')
            progress[disease] = {'first':first,'latest':latest,'diff':diff,'trend':trend}

    all_dates = sorted(set(h['predicted_at'][:10] for h in history))
    colors = {'Diabetes':'#00D2A8','Heart Disease':'#1A8FE3',
              'Liver Disease':'#F39C12','Kidney Disease':'#2ECC71'}
    datasets = []
    for disease in disease_data:
        pts = {h['predicted_at'][:10]: h['probability'] for h in history if h['disease']==disease}
        datasets.append({'label':disease,'data':[pts.get(d) for d in all_dates],'color':colors.get(disease,'#888')})

    chart_data = {'labels': all_dates, 'datasets': datasets}
    high_count = sum(1 for h in history if h['risk_level']=='High Risk')
    low_count  = len(history) - high_count
    conn.close()
    return render_template('patient_detail.html', patient=patient, history=history,
                           progress=progress, chart_data=chart_data,
                           high_count=high_count, low_count=low_count)

@app.route('/save_prediction', methods=['POST'])
def save_prediction():
    conn = get_db()
    conn.execute(
        "INSERT INTO predictions (patient_id, disease, risk_level, probability, input_data, top_factors) VALUES (?,?,?,?,?,?)",
        (request.form['patient_id'], request.form['disease'],
         request.form['risk_level'], float(request.form['probability']),
         request.form.get('input_data','{}'), request.form.get('top_factors','')))
    conn.commit()
    conn.close()
    return redirect(f"/patient/{request.form['patient_id']}")

@app.route('/hospital')
def hospital():
    return render_template('hospital.html')

@app.route('/facedetect')
def facedetect():
    return render_template('facedetect.html')

'''

print(ROUTES_TO_ADD)

from flask import session, redirect, request

# (already have @app.route decorators below — this is just the pattern)

def set_language_route():
    pass  # shown below as proper route

# ── Patient history ───────────────────────────────────────────────

def patients_route():
    conn = get_db()
    rows = conn.execute('SELECT * FROM patients ORDER BY created_at DESC').fetchall()
    # attach latest prediction to each patient
    patients = []
    for p in rows:
        latest = conn.execute(
            'SELECT * FROM predictions WHERE patient_id=? ORDER BY predicted_at DESC LIMIT 1',
            (p['id'],)).fetchone()
        d = dict(p)
        d['latest_risk']    = latest['risk_level']    if latest else None
        d['latest_disease'] = latest['disease']       if latest else None
        patients.append(d)
    conn.close()
    return patients


def patient_detail_route(pid):
    conn = get_db()
    patient = conn.execute('SELECT * FROM patients WHERE id=?', (pid,)).fetchone()
    history = conn.execute(
        'SELECT * FROM predictions WHERE patient_id=? ORDER BY predicted_at DESC',
        (pid,)).fetchall()

    # Build progress comparison per disease
    progress = {}
    disease_data = {}
    for h in reversed(history):   # oldest first
        d = h['disease']
        if d not in disease_data:
            disease_data[d] = []
        disease_data[d].append(h['probability'])

    for disease, probs in disease_data.items():
        if len(probs) >= 2:
            first  = round(probs[0],  1)
            latest = round(probs[-1], 1)
            diff   = round(abs(latest - first), 1)
            if latest < first:
                trend = 'improved'
            elif latest > first:
                trend = 'worsened'
            else:
                trend = 'stable'
            progress[disease] = {'first':first, 'latest':latest, 'diff':diff, 'trend':trend}

    # Chart data
    all_dates = sorted(set(h['predicted_at'][:10] for h in history))
    disease_colors = {
        'Diabetes':'#00D2A8', 'Heart Disease':'#1A8FE3',
        'Liver Disease':'#F39C12', 'Kidney Disease':'#2ECC71'
    }
    chart_datasets = []
    for disease in disease_data:
        pts = {}
        for h in history:
            if h['disease'] == disease:
                pts[h['predicted_at'][:10]] = h['probability']
        chart_datasets.append({
            'label':  disease,
            'data':   [pts.get(d) for d in all_dates],
            'color':  disease_colors.get(disease, '#888888'),
        })

    high_count = sum(1 for h in history if h['risk_level'] == 'High Risk')
    low_count  = sum(1 for h in history if h['risk_level'] == 'Low Risk')

    conn.close()
    return patient, history, progress, {
        'labels': all_dates, 'datasets': chart_datasets
    }, high_count, low_count

if __name__ == '__main__':
    app.run(debug=True)