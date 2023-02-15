from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
import joblib
import json
import ast
import numpy as np
import pandas as pd
import category_encoders as ce

app = Flask(__name__)
app.config["DEBUG"] = True
api = Api(app)

# load trained model
knn_model = joblib.load("./trained/trainedmodel.sav")
encoder5 = joblib.load("./trained/encoderfile.sav")
poly = joblib.load("./trained/polyfile.sav")


@app.route('/')
def index():
    return {'msg':"helllo world, you're accessing /"},200

@app.route("/compute", methods=['POST'])
def process_json():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'text/plain'):
        HRinput = request.data
        dict_str = HRinput.decode("UTF-8")
        mydata = ast.literal_eval(dict_str)
        print(mydata)
        #return json

        # Process the input
        HRinput = pd.DataFrame(mydata)
        HRinputEnco = encoder5.transform(HRinput)
        HRpoly_features = poly.transform(HRinputEnco)
        HRinputSalaryPred = knn_model.predict(HRpoly_features)

        print(HRinputSalaryPred)
        return {'result': HRinputSalaryPred[0]}, 200
        #return {'result': "ok"}, 200
    else:
        return 'Content-Type not supported!'

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')  # run our Flask app