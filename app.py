from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/income_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        age = int(request.form['age'])
        education_num = int(request.form['education_num'])
        hours = int(request.form['hours'])
        occupation = request.form['occupation']
        marital_status = request.form['marital_status']
        sex = request.form['sex']

        # Simplificación: codificación dummy simulada
        features = np.array([[age, education_num, hours, 1 if sex == 'Male' else 0]])
        prediction = model.predict(features)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
