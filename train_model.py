import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# ✅ Cargar los datasets
caracteristicas_path = 'model/CSV/Características generales, seguridad social en salud y educación.CSV'
ocupados_path = 'model/CSV/Ocupados.CSV'
fuerza_trabajo_path = 'model/CSV/Fuerza de trabajo.CSV'

caracteristicas = pd.read_csv(caracteristicas_path, sep=';', encoding='latin1')
ocupados = pd.read_csv(ocupados_path, sep=';', encoding='latin1')
fuerza_trabajo = pd.read_csv(fuerza_trabajo_path, sep=';', encoding='latin1')

# Unir datasets usando las claves comunes
df = caracteristicas.merge(
    ocupados[['DIRECTORIO', 'SECUENCIA_P', 'ORDEN', 'INGLABO']],
    on=['DIRECTORIO', 'SECUENCIA_P', 'ORDEN'],
    how='inner'
).merge(
    fuerza_trabajo[['DIRECTORIO', 'SECUENCIA_P', 'ORDEN', 'P6240']],  # P6240: Horas trabajadas reales
    on=['DIRECTORIO', 'SECUENCIA_P', 'ORDEN'],
    how='left'
)

# Filtrar variables de interés
columns_needed = ['P6040', 'P6080', 'P6240', 'INGLABO']  # Edad, Educación, Horas Trabajadas, Ingreso
df = df[columns_needed]
df.dropna(inplace=True)
df = df[df['INGLABO'] > 0]

# Renombrar columnas
df.columns = ['age', 'education_num', 'hours_per_week', 'income_amount']

# Asegurar que las horas de trabajo sean numéricas y razonables
df['hours_per_week'] = pd.to_numeric(df['hours_per_week'], errors='coerce').fillna(40).clip(upper=80)

# Ingeniería de características: Ajuste de pesos para balancear la importancia
df['education_num'] = df['education_num'] * 3
df['hours_per_week'] = df['hours_per_week'] * 3 

# Variables de entrada y salida
X = df[['age', 'education_num', 'hours_per_week']]
y = df['income_amount']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo
output_dir = 'model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'income_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

print("✅ Modelo entrenado y guardado como 'income_model.pkl'")

# Mostrar la nueva importancia de las variables
importances = model.feature_importances_
feature_names = ['age', 'education_num', 'hours_per_week']

print("\n Importancia de las Variables en el Modelo:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")
