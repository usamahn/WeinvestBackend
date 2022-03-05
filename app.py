import pandas as pd
from flask import Flask, request,jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)





@app.route('/')
def home():
    return("hello world")

"""
@app.route('/predict', methods=['POST'])
def predict():
    col = ['cif_value',
           'country',
           'date',
           'fob_value',
           'gross_weight',
           'id',
           'importer_id',
           'office_id',
           'quantity',
           'tariff_code',
           'total_taxes']
    d = dict((request.form))
    df = pd.DataFrame(columns=col)
    df=df.append(d,ignore_index=True)
    df=process_data(df)
    df.columns=['importer_id', 'country', 'office_id', 'tariff_code', 'quantity', 'gross_weight', 'fob_value', 'cif_value', 'total_taxes']
    y_pred=model.predict_proba(df.astype(int))
    response = jsonify({'prob': str(y_pred[:,1][0])})
    return response"""


if __name__ == "__main__":
    #port = int(os.environ.get('PORT', 8000))
    app.run(debug=True)
