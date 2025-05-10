import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Cargar el dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
            'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
            'hours_per_week', 'native_country', 'income']

df = pd.read_csv(url, header=None, names=columns, na_values=' ?', skipinitialspace=True)
df.dropna(inplace=True)  # Eliminar valores nulos

# Convertir ingresos de USD a COP (Simulación de ingresos > 200 millones COP)
df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Selección de variables
X = df[['age', 'education_num', 'hours_per_week', 'sex']]
y = df['income']

# Codificar variable 'sex'
le = LabelEncoder()
X.loc[:, 'sex'] = le.fit_transform(X['sex'])

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo
with open('model/income_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo entrenado y guardado como 'income_model.pkl'")
