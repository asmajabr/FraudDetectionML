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
#post a json object to see the prediction for it :
@app.route('/', methods=['GET'])
def info():
    result = [
        {
            'Author' : 'DS team',
            'description' : 'A fraud detection model using a kaggle dataset',
        }
    ]
    return jsonify(result)  
@app.route('/verify', methods=['GET','POST'])
def predict_new_transaction():
    loaded_model = joblib.load(filename).set_params(n_jobs=1)
    
    json_ = request.json
    query_df = pd.DataFrame([json_])
    prediction = str(loaded_model.predict(query_df))
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=False, host="127.0.0.1", threaded=False)

