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
    # Simular datos de ejemplo (puedes reemplazar con datos reales)
    np.random.seed(42)
    edad = np.random.randint(18, 65, 100)
    ingresos = 1000000 + edad * 50000 + np.random.normal(0, 500000, 100)

    X = edad.reshape(-1, 1)
    y = ingresos

    modelo = LinearRegression()
    modelo.fit(X, y)
    predicciones = modelo.predict(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(edad, ingresos, label="Datos reales", alpha=0.6)
    plt.plot(edad, predicciones, color="red", label="Línea de regresión", linewidth=2)
    plt.xlabel("Edad")
    plt.ylabel("Ingreso estimado (COP)")
    plt.title("Relación entre Edad e Ingreso Estimado")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Guardar la gráfica en static
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
        sex = 1 if request.form['sex'] == 'Male' else 0

        features = pd.DataFrame([[age, education_num, hours, sex]], columns=['age', 'education_num', 'hours_per_week', 'sex'])
        prediction = model.predict(features)[0]

        # Gráfica comparativa
        average_income = 20_000_000  # Promedio nacional anual (ajustable)
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

        # Nueva gráfica: Distribución de Ingresos por Nivel Educativo
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

        # Generar la nueva gráfica de dispersión
        generar_grafica_dispersion()

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
