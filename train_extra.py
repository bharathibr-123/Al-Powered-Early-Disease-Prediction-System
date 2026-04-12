import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

le = LabelEncoder()

# ── LIVER ──────────────────────────────────────────────
df_liver = pd.read_csv('liver.csv')
df_liver.dropna(inplace=True)
if 'Gender' in df_liver.columns:
    df_liver['Gender'] = le.fit_transform(df_liver['Gender'])
target_liver = 'Dataset' if 'Dataset' in df_liver.columns else df_liver.columns[-1]
X_l = df_liver.drop(target_liver, axis=1)
y_l = df_liver[target_liver]
X_train, X_test, y_train, y_test = train_test_split(X_l, y_l, test_size=0.2, random_state=42)
liver_model = RandomForestClassifier(n_estimators=100, random_state=42)
liver_model.fit(X_train, y_train)
print(f"Liver Accuracy: {accuracy_score(y_test, liver_model.predict(X_test))*100:.2f}%")
with open('liver_model.pkl','wb') as f:
    pickle.dump(liver_model, f)
print("liver_model.pkl saved!")

# ── KIDNEY ─────────────────────────────────────────────
df_kidney = pd.read_csv('kidney.csv')

# Drop id column, strip spaces from column names
df_kidney.drop(columns=['id'], inplace=True, errors='ignore')
df_kidney.columns = df_kidney.columns.str.strip()  # removes trailing spaces like 'bu '

# Replace ? with NaN and drop
df_kidney.replace('?', float('nan'), inplace=True)
df_kidney.dropna(inplace=True)

# Encode all text columns
for col in df_kidney.select_dtypes(include='object').columns:
    df_kidney[col] = le.fit_transform(df_kidney[col].astype(str))

# Save cleaned column list (excluding target)
target_kidney = 'classification'
X_k = df_kidney.drop(target_kidney, axis=1)
y_k = df_kidney[target_kidney]

# Save feature names for later use
print("Kidney features used:", list(X_k.columns))
print("Number of features:", len(X_k.columns))

X_train, X_test, y_train, y_test = train_test_split(X_k, y_k, test_size=0.2, random_state=42)
kidney_model = RandomForestClassifier(n_estimators=100, random_state=42)
kidney_model.fit(X_train, y_train)
print(f"Kidney Accuracy: {accuracy_score(y_test, kidney_model.predict(X_test))*100:.2f}%")
with open('kidney_model.pkl','wb') as f:
    pickle.dump(kidney_model, f)
print("kidney_model.pkl saved!")