from flask import Flask
from flask import request
from flask import jsonify
import pickle
import numpy as np


model_name = 'rs-churn-model.bin'

## To load the model
with open(model_name, 'rb') as f_in:
    dv,model = pickle.load(f_in)

app = Flask('rs-churn-service')

@app.route('/predict', methods = ['POST'])
def predict():
    customer = request.get_json()
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    
    churn = y_pred >= 0.5
    
    result = {
        'churn-probablity': float(y_pred),
        'churn': bool(churn)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=9696)





# churn_prob = pred_one(customer, dv, model)
# print(f"The input customer is: \n {customer}")
# print(f"\n The Churn probability is: {churn_prob}")

# url = 'http://localhost:9696/predict'
# response = requests.post(url, json=customer)
# result = response.json()
# result
