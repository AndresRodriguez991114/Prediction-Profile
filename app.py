import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
model = pickle.load(open('model/income_model.pkl', 'rb'))

def generar_grafica_dispersion():
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle
    import numpy as np
    import os

    # Cargar el modelo entrenado
    model_path = 'model/income_model.pkl'
    if not os.path.exists(model_path):
        print("⚠️ Modelo no encontrado en la ruta especificada.")
        return

    model = pickle.load(open(model_path, 'rb'))

    # Cargar datos reales desde Características Generales
    caracteristicas_path = 'model/CSV/Características generales, seguridad social en salud y educación.CSV'
    if not os.path.exists(caracteristicas_path):
        print("⚠️ Dataset no encontrado en la ruta especificada.")
        return

    caracteristicas = pd.read_csv(caracteristicas_path, sep=';', encoding='latin1')

    # Verificar existencia de columnas necesarias
    if not all(col in caracteristicas.columns for col in ['P6040', 'P6080']):
        print("⚠️ Las columnas P6040 (Edad) y P6080 (Nivel Educativo) no están en el dataset.")
        return

    # Preparar los datos
    df = caracteristicas[['P6040', 'P6080']].dropna()
    df = df[(df['P6040'] >= 18) & (df['P6040'] <= 65)]  # Filtrar edades entre 18 y 65

    if df.empty:
        print("⚠️ No hay suficientes datos para generar la gráfica de dispersión.")
        return

    # Agregar columnas faltantes requeridas por el modelo
    df['hours_per_week'] = 40  # Asumiendo promedio de 40 horas semanales

    # Preparar entrada para el modelo
    X = df[['P6040', 'P6080', 'hours_per_week']]
    X.columns = ['age', 'education_num', 'hours_per_week']

    # Generar predicciones
    ingresos_predichos = model.predict(X)

    # Crear la gráfica de dispersión
    plt.figure(figsize=(8, 5))
    plt.scatter(df['P6040'], ingresos_predichos, alpha=0.5, color='skyblue', label="Predicciones Reales")
    plt.xlabel("Edad")
    plt.ylabel("Ingreso Estimado (COP)")
    plt.title("Relación entre Edad e Ingreso Estimado (GEIH)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Guardar la gráfica en la carpeta static
    plt.savefig('static/grafica_dispersion.png')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        age = int(request.form['age'])
        education_input = request.form['education_num']
        if education_input == "":
            return render_template('index.html', prediction=None)
        education_num = int(education_input)
        hours = int(request.form['hours'])

        features = pd.DataFrame([[age, education_num, hours]], columns=['age', 'education_num', 'hours_per_week'])
        prediction = model.predict(features)[0] * 12 

        # Gráfica comparativa
        average_income = 25_000_000  # Promedio anual 
        categories = ['Promedio Nacional', 'Tu Predicción']
        values = [average_income, prediction]

        fig, ax = plt.subplots()
        bars = ax.bar(categories, values, color=['gray', 'green'])
        ax.set_ylabel('Pesos Colombianos')
        ax.set_title('Comparación de Ingresos')

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f"${int(yval):,}", 
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('static/prediction_plot.png')
        plt.close()

        # Gráfica de Ingresos por Nivel Educativo
        education_levels = ['Bachillerato', 'Pregrado', 'Especialización', 'Maestría', 'Doctorado']
        incomes = [18_000_000, 36_000_000, 60_000_000, 84_000_000, 120_000_000]

        fig2, ax2 = plt.subplots()
        bars = ax2.bar(education_levels, incomes, color='skyblue')
        ax2.set_ylabel('Pesos Colombianos')
        ax2.set_title('Ingresos Promedio por Nivel Educativo')

        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval, f"${int(yval):,}", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('static/education_plot.png')
        plt.close()

        #Gráfica de dispersión
        generar_grafica_dispersion()

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
