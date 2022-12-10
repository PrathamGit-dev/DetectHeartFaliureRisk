# from importlib_metadata import NullFinder
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import pandas as pd
import sklearn

app = Flask(__name__)
model = pickle.load(open('model_rfc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html',prediction_text=" ")

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    featre_index=[y for y in request.form.keys()]
    # print(int_features)
    # print("keys->",featre_index)
    output=prediction_final(int_features)
    message=" "
    # print("output is->",output)
    color="green"
    if output==0:
        message="You Are Safe"
        color="green"
    else:
        message="You need to consult a doctor"
        color="red"
    
    if(int_features[0] is None):
        return redirect("/")
    else:
        return render_template('index2.html', prediction_text=message,colour=color)

def prediction_final(feats):
    df_empty=pd.DataFrame(columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])
    to_append=[float(feats[0]), feats[1], feats[2], float(feats[3]),float(feats[4]), float(feats[5]), feats[6], float(feats[7]), feats[8],float(feats[9]),feats[10]]#create only this array
    a_series = pd. Series(to_append, index = df_empty. columns)
    df = df_empty. append(a_series, ignore_index=True)
    prediction = model.predict(df)
    return prediction

if __name__ == "__main__":
    app.run(debug=True)