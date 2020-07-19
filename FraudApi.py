from flask import request, jsonify
import numpy
import json
import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn.externals
import joblib
from flask import Flask
app = Flask(__name__)
filename = 'tuned_rf_smote.sav'
loaded_model = joblib.load(filename)
def pred_vect(v):
    vect = {}
    vect["Time"]=v[0]
    j = 1
    for i in range (1,28):
        ind = "V"+str(j)
        vect [ind] = v[i]
        j = j+1
    vect ["Amount"] = v[28]
    return pd.DataFrame([vect])
#v = [4462,-2.303349568,1.75924746,-0.359744743,2.330243051,-0.821628328,-0.075787571,0.562319782,-0.399146578,-0.238253368,-1.525411627,2.032912158,-6.560124295,0.022937323,-1.470101536,-0.698826069,-2.282193829,-4.781830856,-2.615664945,-1.334441067,-0.430021867,-0.294166318,-0.932391057,0.172726296,-0.087329538,-0.156114265,-0.542627889,0.039565989,-0.153028797,239.93]
v=[4462,-2.303349568,1.75924746,-0.359744743,2.330243051,-0.821628328,-0.075787571,0.562319782,-0.399146578,-0.238253368,-1.525411627,2.032912158,-6.560124295,0.022937323,-1.470101536,-0.698826069,-2.282193829,-4.781830856,-2.615664945,-1.334441067,-0.430021867,-0.294166318,-0.932391057,0.172726296,-0.087329538,-0.156114265,-0.542627889,0.039565989,-0.153028797,239.93]
#post a json object to see the prediction for it :
@app.route('/chkreq', methods=['POST']) 
def foo():
    print (request)
    #return request
    return json.dumps(request.json)
@app.route('/postjson', methods = ['POST'])
def postJsonHandler():
    print (request.is_json)
    content = request.get_json()
    print (content)
    return jsonify(content)
@app.route('/', methods=['GET'])
def home():
    result = [
        {
            'Author' : 'Fraud Detection Team',
            'description' : 'A tuned fraud detection model using a kaggle dataset',
        }
    ]
    return jsonify(result)
@app.route('/verify', methods=['GET','POST'])
def predict_new_transaction():
    input = pd.DataFrame([request.json])
    pred_ = str(loaded_model.predict(input))
    result = [{'prediction': pred_} ]
    return jsonify(pred_)
    
@app.route('/api/v0/test', methods=['GET','POST'])
def predict_test():
    loaded_model = joblib.load(filename)
    res = str(loaded_model.predict(pred_vect(v))[0])
    result = [
    {'id': 0,
     'prediction': res} ]
    print(res) 
    return jsonify(result)
@app.route('/api/v0/info', methods=['GET'])
def info():
    result = [
        {
            'Author' : 'Fraud Detection Team',
            'description' : 'A tuned fraud detection model using a kaggle dataset',
        }
    ]
    return jsonify(result)  

if __name__ == '__main__':
    app.run(debug=True)
