from flask import Flask, render_template, request, jsonify
from flask_restful import Resource, Api, reqparse
import os
#from os.path import join, dirname, realpath
import joblib
import json
import ast
import numpy as np
import pandas as pd
import category_encoders as ce

app = Flask(__name__)
app.config["DEBUG"] = True
#api = Api(app)
ALLOWED_EXTENSIONS = {'csv'}

# load trained model
knn_model = joblib.load("./trained/trainedmodel.sav")
encoder5 = joblib.load("./trained/encoderfile.sav")
poly = joblib.load("./trained/polyfile.sav")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
   return render_template('index.html')

@app.route("/upload", methods=['POST'])
def process_json():
    if 'fname' not in request.files:
        return {'msg': 'File Not Found'},400

    file = request.files['fname']

    if file and allowed_file(file.filename):

        filename = file.filename

        # init the fullpath variable
        fullpath = os.path.join('./tmp',filename)

        # save file to path
        file.save(fullpath)

        # Processing the CSV
        result = processCSV(fullpath)
        return render_template('index.html',data=result)
    else:
        return 'Content-Type not supported!'
def processCSV(path):
    df = pd.read_csv(path,sep=',')
    #dict_str = HRinput.decode("UTF-8")
    #mydata = ast.literal_eval(dict_str)
    print(df)
    #return json

    # Process the input
    #HRinput = pd.DataFrame(mydata)
    #HRinputEnco = encoder5.transform(HRinput)
    HRinputEnco = encoder5.transform(df)
    HRpoly_features = poly.transform(HRinputEnco)
    HRinputSalaryPred = knn_model.predict(HRpoly_features)

    #print(HRinputSalaryPred)
    return HRinputSalaryPred
    #return df.loc[:,'Salary'].mean(),df.loc[:,'Salary'].max(),df.loc[:,'Salary'].min()

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')  # run our Flask app