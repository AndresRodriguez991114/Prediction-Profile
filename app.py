from flask import Flask, render_template, request
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

        features = np.array([[age, education_num, hours, sex]])
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

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
