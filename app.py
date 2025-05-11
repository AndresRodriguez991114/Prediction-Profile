import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = pickle.load(open('model/income_model.pkl', 'rb'))

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
        average_income = 100_000_000  # Promedio nacional anual (ajustable)
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
        incomes = [18_115_440, 36_000_000, 60_000_000, 84_000_000, 120_000_000]

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


    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
