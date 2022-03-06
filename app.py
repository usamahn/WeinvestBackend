import pandas as pd
from flask import Flask, request,jsonify
import pickle
from flask_cors import CORS
import pypfopt
from pypfopt import risk_models
from pypfopt import plotting
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import objective_functions
import investpy


app = Flask(__name__)
CORS(app)


@app.route('/getAnnualReturn', methods=['POST'])
def getAnnualReturn():
  d = request.json
  print(d["risk"])
  risk=0.25
  if(d["risk"]=="high"):
    risk=0.75
  elif(d["risk"]=="medium"):
    risk=0.40
  elif(d["risk"]=="low"):
    risk=0.25
  print("=========================================")
  aapl = investpy.get_stock_historical_data(stock='AAPL',country='United States',from_date='01/01/2018',to_date='01/03/2022')
  msft = investpy.get_stock_historical_data(stock='MSFT',country='United States',from_date='01/01/2018',to_date='01/03/2022')
  tsla = investpy.get_stock_historical_data(stock='TSLA',country='United States',from_date='01/01/2018',to_date='01/03/2022')
  amzn = investpy.get_stock_historical_data(stock='AMZN',country='United States',from_date='01/01/2018',to_date='01/03/2022')
  btc = investpy.get_crypto_historical_data(crypto='bitcoin',from_date='01/01/2018',to_date='01/03/2022')
  eth = investpy.get_crypto_historical_data(crypto='ethereum',from_date='01/01/2018',to_date='01/03/2022')
  ada=investpy.get_crypto_historical_data(crypto='cardano',from_date='01/01/2018',to_date='01/03/2022')
  bond3y=investpy.get_bond_historical_data('U.S. 3Y',from_date='01/01/2018',to_date='01/03/2022')
  bond5y=investpy.get_bond_historical_data('U.S. 5Y',from_date='01/01/2018',to_date='01/03/2022')
  bond10y=investpy.get_bond_historical_data('U.S. 10Y',from_date='01/01/2018',to_date='01/03/2022')
  prices = pd.DataFrame({'aapl': pd.DataFrame.reset_index(aapl).iloc[:,4], 'msft':pd.DataFrame.reset_index(msft).iloc[:,4],
                   'tsla': pd.DataFrame.reset_index(tsla).iloc[:,4],'amzn': pd.DataFrame.reset_index(amzn).iloc[:,4],
                    'btc': pd.DataFrame.reset_index(btc).iloc[:,4], 'eth': pd.DataFrame.reset_index(eth).iloc[:,4],
                     'ada': pd.DataFrame.reset_index(ada).iloc[:,4],'bond3y': pd.DataFrame.reset_index(bond3y).iloc[:,4],
                     'bond5y': pd.DataFrame.reset_index(bond5y).iloc[:,4],'bond10y': pd.DataFrame.reset_index(bond10y).iloc[:,4]})
  prices=prices.dropna()
  S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
  mu = expected_returns.capm_return(prices)
  ef = EfficientFrontier(mu, S)
  ef.add_objective(objective_functions.L2_reg, gamma=0.1)
  ef.efficient_risk(target_volatility=risk)
  weights = ef.clean_weights()
  res= {
    "Stocks": weights['aapl'] + weights['msft']+weights['tsla']+weights['amzn'],
    "Cryptocurrencies": weights['eth'] + weights['ada']+weights['btc'],
    "Bonds": weights['bond3y'] + weights['bond5y']+weights['bond10y'],
    "Annual Return" : ef.portfolio_performance()[0]
  }
  return(res)


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
