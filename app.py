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
        education_num = int(request.form['education_num'])
        hours = int(request.form['hours'])
        sex = 1 if request.form['sex'] == 'Male' else 0

        features = np.array([[age, education_num, hours, sex]])
        prediction = model.predict(features)[0]

        # Generar gráfica de predicción
        fig, ax = plt.subplots()
        bars = ['Ingreso Aproximado']
        values = [prediction]
        ax.bar(bars, values)
        ax.set_ylabel('Pesos Colombianos')
        ax.set_title('Resultado de la Predicción')
        plt.savefig('static/prediction_plot.png')
        plt.close()

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
