from flask import Flask, request, jsonify
import pickle
import sys
import os
import numpy as np
import sklearn
# --------------------
model_rf = pickle.load(open('model_rf.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def home():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    N = request.form.get('N')
    P = request.form.get('P')
    K = request.form.get('K')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')

    input_query = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    result = model_rf.predict(input_query)[0]

    # result = {'N':N,'P':P,'K':K,'temperature':temperature,'humidity':humidity,'ph':ph,'rainfall':rainfall }
    return jsonify({'Result':result})

if __name__=='__main__':
    app.run(debug=True)



