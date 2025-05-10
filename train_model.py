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
df.dropna(inplace=True)  # Eliminar registros con valores nulos

# Convertir la columna de ingresos a una cantidad aproximada en COP
# Si gana >50K USD, asumimos un ingreso promedio de 250 millones COP, si no, 50 millones COP
df['income_amount'] = df['income'].apply(lambda x: 250_000_000 if x.strip() == '>50K' else 50_000_000)

# Selección de variables importantes
X = df[['age', 'education_num', 'hours_per_week', 'sex']]
y = df['income_amount']

# Codificación de la variable 'sex'
le = LabelEncoder()
X.loc[:, 'sex'] = le.fit_transform(X['sex'])

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo con Random Forest Regressor (predicción de cantidad)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo entrenado
with open('model/income_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Modelo entrenado y guardado como 'income_model.pkl'")
