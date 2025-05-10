from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
model = pickle.load(open('model/income_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Obtener los datos del formulario
        age = int(request.form['age'])
        education_num = int(request.form['education_num'])
        hours = int(request.form['hours'])
        sex = 1 if request.form['sex'] == 'Male' else 0

        # Preparar los datos para el modelo
        features = np.array([[age, education_num, hours, sex]])
        prediction = model.predict(features)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
