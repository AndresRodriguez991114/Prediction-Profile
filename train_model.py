
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# ✅ Cargar el dataset desde el archivo local de la GEIH (Ocupados)
file_path = 'model/CSV/Ocupados.CSV'
df = pd.read_csv(file_path, sep=';', encoding='latin1')

# ✅ Filtrar las columnas necesarias
# P6040: Edad, P6080: Nivel Educativo, P6240: Horas trabajadas semana, P6020: Sexo, INGLABO: Ingreso mensual
columns_needed = ['P6040', 'P6080', 'P6240', 'P6020', 'INGLABO']
df = df[columns_needed]
df.dropna(inplace=True)
df = df[df['INGLABO'] > 0]  # Eliminar registros sin ingreso

# ✅ Renombrar las columnas para facilitar su uso
df.columns = ['age', 'education_num', 'hours_per_week', 'sex', 'income_amount']

# ✅ Codificar 'sex' (Asumiendo: 1 = Hombre, 2 = Mujer)
df['sex'] = df['sex'].map({1: 1, 2: 0}).fillna(0).astype(int)

# ✅ Variables de entrada y salida
X = df[['age', 'education_num', 'hours_per_week', 'sex']]
y = df['income_amount']

# ✅ Codificar 'sex' si aún no está como número (ya se hizo antes, pero mantenemos por si acaso)
le = LabelEncoder()
X.loc[:, 'sex'] = le.fit_transform(X['sex'])

# ✅ Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Entrenar el modelo de regresión
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Guardar el modelo entrenado
output_dir = 'model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'income_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

print("✅ Modelo entrenado con datos de Ocupados y guardado como 'income_model.pkl'")
