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

# Ajustar ingresos estimados según nivel educativo
education_income = {
    9: 18_000_000,     # Bachillerato (ingreso anual mínimo)
    13: 36_000_000,    # Pregrado (estimado anual ~3 millones/mes)
    15: 60_000_000,    # Especialización (~5 millones/mes)
    16: 84_000_000,    # Maestría (~7 millones/mes)
    17: 120_000_000    # Doctorado (~10 millones/mes)
}
# Si no está en el mapa, asigna ingreso base de 12 SMLV
df['income_amount'] = df['education_num'].map(education_income).fillna(18_000_000)

# Variables de entrada y salida
X = df[['age', 'education_num', 'hours_per_week', 'sex']]
y = df['income_amount']

# Codificar variable categórica 'sex'
le = LabelEncoder()
X.loc[:, 'sex'] = le.fit_transform(X['sex'])

# Dividir dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo de regresión
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo entrenado
with open('model/income_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Modelo entrenado y guardado como 'income_model.pkl'")
