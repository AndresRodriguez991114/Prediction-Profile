import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

#  Cargar los datasets
caracteristicas_path = 'model/CSV/Características generales, seguridad social en salud y educación.CSV'
ocupados_path = 'model/CSV/Ocupados.CSV'
Fuerza_de_trabajo_path = 'model/CSV/Fuerza de trabajo.CSV'

caracteristicas = pd.read_csv(caracteristicas_path, sep=';', encoding='latin1')
ocupados = pd.read_csv(ocupados_path, sep=';', encoding='latin1')
Fuerza_de_trabajo = pd.read_csv(Fuerza_de_trabajo_path, sep=';', encoding='latin1')

#  Unir datasets
df = caracteristicas.merge(
    ocupados[['DIRECTORIO', 'SECUENCIA_P', 'ORDEN', 'INGLABO']],
    on=['DIRECTORIO', 'SECUENCIA_P', 'ORDEN'],
    how='inner'
)

#  Filtrar variables de interés
columns_needed = ['P6040', 'P6080', 'INGLABO']  # Edad, Educación, Ingreso
df = df[columns_needed]
df.dropna(inplace=True)
df = df[df['INGLABO'] > 0]

#  Renombrar columnas
df.columns = ['age', 'education_num', 'income_amount']

#  Agregar 'hours_per_week' asumido como promedio de 40 horas
df['hours_per_week'] = 40

#  Variables de entrada y salida
X = df[['age', 'education_num', 'hours_per_week']]
y = df['income_amount']

#  División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Entrenar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Guardar el modelo
output_dir = 'model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'income_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

print(" Modelo entrenado y guardado como 'income_model.pkl'")
