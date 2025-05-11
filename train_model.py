import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Cargar el dataset desde UCI
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
           'hours_per_week', 'native_country', 'income']

df = pd.read_csv(url, header=None, names=columns, na_values=' ?', skipinitialspace=True)
df.dropna(inplace=True)

# Ajustar ingresos estimados según nivel educativo y factor de horas trabajadas
education_income = {
    9: 1_509_620 * 12,      # Bachillerato (salario mínimo anual)
    13: 3_000_000 * 12,     # Pregrado (~3M mensual)
    15: 5_000_000 * 12,     # Especialización
    16: 7_000_000 * 12,     # Maestría
    17: 10_000_000 * 12     # Doctorado
}

# Asignar ingreso base según nivel educativo
df['base_income'] = df['education_num'].map(education_income).fillna(1_509_620 * 12)

# Ajustar ingresos según horas de trabajo (más horas, más ingreso)
df['income_amount'] = df['base_income'] * (df['hours_per_week'] / 40).clip(upper=1.5)

# Variables de entrada y salida
X = df[['age', 'education_num', 'hours_per_week', 'sex']]
y = df['income_amount']

# Codificar 'sex'
le = LabelEncoder()
X.loc[:, 'sex'] = le.fit_transform(X['sex'])

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo de regresión
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo entrenado
with open('model/income_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Modelo entrenado y guardado como 'income_model.pkl'")
