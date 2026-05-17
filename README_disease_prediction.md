<div align="center">

# 🏥 AI-Powered Early Disease Prediction System

### Predict. Prevent. Protect.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org)
[![HTML](https://img.shields.io/badge/HTML-Frontend-E34F26?style=flat-square&logo=html5&logoColor=white)](.)
[![Diseases](https://img.shields.io/badge/Diseases%20Covered-4-blueviolet?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br/>

> **Detect the risk of Diabetes, Heart Disease, Kidney Disease & Liver Disease — before symptoms appear.**

</div>

---

## 📌 Problem Statement

Millions of people suffer from chronic diseases that could have been prevented or managed if caught early. Most patients only seek diagnosis after symptoms become severe — by which time treatment is far more complex and costly.

> Early detection saves lives. But access to fast, affordable screening is limited.

**This system brings AI-powered early disease screening directly to the user.**

---

## 💡 Solution

An intelligent, multi-disease prediction web application that takes patient health metrics as input and instantly predicts the likelihood of **4 major diseases** using trained Machine Learning models — all from a simple web interface.

---

## 🦠 Diseases Covered

| Disease | Dataset | Trained Model |
|---|---|---|
| 🩸 **Diabetes** | `diabetes.csv` | `model.pkl` |
| ❤️ **Heart Disease** | `heart.csv` | `heart_model.pkl` |
| 🫘 **Kidney Disease** | `kidney.csv` | `kidney_model.pkl` |
| 🫀 **Liver Disease** | `liver.csv` | `liver_model.pkl` |

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **4 ML Prediction Models** | Separate trained models for Diabetes, Heart, Kidney & Liver disease |
| ⚡ **Instant Risk Prediction** | Enter health metrics and get a prediction in seconds |
| 🌐 **Web-Based Interface** | Clean HTML frontend — no installation needed for end users |
| 🗄️ **SQLite Database** | Persistent storage of user data and prediction history via `healthai.db` |
| 🔬 **Multiple Training Scripts** | Modular training pipeline — `model.py`, `train_heart.py`, `train_extra.py`, `check_kidney.py` |
| 📁 **Organized Templates** | Dedicated HTML templates for each disease and result page |

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Web Framework** | Flask |
| **ML Models** | Scikit-learn (saved as `.pkl`) |
| **Frontend** | HTML5 · CSS3 |
| **Database** | SQLite (`healthai.db`) |
| **Data** | CSV datasets (Diabetes, Heart, Kidney, Liver) |

</div>

---

## 📂 Project Structure

```
Al-Powered-Early-Disease-Prediction-System/
│
├── 📁 templates/               # HTML frontend pages
│   ├── index.html              # Home / landing page
│   ├── diabetes.html           # Diabetes prediction form
│   ├── heart.html              # Heart disease prediction form
│   ├── kidney.html             # Kidney disease prediction form
│   ├── liver.html              # Liver disease prediction form
│   └── result.html             # Prediction result display
│
├── app.py                      # Main Flask application & routes
│
├── model.py                    # Diabetes model training script
├── train_heart.py              # Heart disease model training script
├── train_extra.py              # Additional model training utilities
├── check_kidney.py             # Kidney model validation script
│
├── model.pkl                   # Trained Diabetes prediction model
├── heart_model.pkl             # Trained Heart Disease model
├── kidney_model.pkl            # Trained Kidney Disease model
├── liver_model.pkl             # Trained Liver Disease model
│
├── diabetes.csv                # Diabetes training dataset
├── heart.csv                   # Heart disease training dataset
├── kidney.csv                  # Kidney disease training dataset
├── liver.csv                   # Liver disease training dataset
│
├── healthai.db                 # SQLite database (user & prediction records)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/bharathibr-123/Al-Powered-Early-Disease-Prediction-System.git
cd Al-Powered-Early-Disease-Prediction-System
```

**2. Install dependencies**
```bash
pip install flask scikit-learn pandas numpy
```

> Or if a `requirements.txt` is available:
> ```bash
> pip install -r requirements.txt
> ```

**3. (Optional) Retrain the models**
```bash
# Retrain diabetes model
python model.py

# Retrain heart disease model
python train_heart.py

# Retrain additional models
python train_extra.py

# Validate kidney model
python check_kidney.py
```

**4. Run the Flask app**
```bash
python app.py
```

**5. Open your browser**
```
http://localhost:5000
```

---

## 🧠 How It Works

```
User Inputs Health Metrics
        │
        ▼
   Flask Web App (app.py)
        │
        ▼
  Select Disease to Predict
        │
        ├──── Diabetes  ──→  model.pkl
        ├──── Heart     ──→  heart_model.pkl
        ├──── Kidney    ──→  kidney_model.pkl
        └──── Liver     ──→  liver_model.pkl
                │
                ▼
     ML Model Processes Input
                │
                ▼
      Risk Prediction Result
      (Positive / Negative)
                │
                ▼
    Result Displayed on Web Page
                │
                ▼
    Data Saved to healthai.db
```

---

## 📋 Input Features (Example — Diabetes)

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index |
| Diabetes Pedigree | Diabetes pedigree function score |
| Age | Age in years |

> Each disease module has its own relevant set of clinical input features.

---

## 🎯 Use Cases

- 🏥 **Hospitals & Clinics** — Quick preliminary screening before full diagnosis
- 👨‍👩‍👧 **Individuals** — Self-assessment tool for understanding health risk factors
- 🎓 **Medical Students** — Learning how ML is applied in real clinical scenarios
- 🔬 **Researchers** — Baseline multi-disease ML prediction framework

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## ⚠️ Disclaimer

> This application is built for **educational and research purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider for medical decisions.

---

## 👤 Author

**Bharathi BR**

[![GitHub](https://img.shields.io/badge/GitHub-bharathibr--123-181717?style=flat-square&logo=github)](https://github.com/bharathibr-123)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ If this project helped or inspired you, please give it a star!

*"AI in healthcare isn't about replacing doctors — it's about empowering people with knowledge before it's too late."*

**Made with ❤️ | Building solutions that matter. 🚀**

</div>
