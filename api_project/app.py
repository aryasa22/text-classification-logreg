import numpy as np 
from flask import Flask, request, jsonify, render_template
from flask_restx import Api, Resource, fields
import pickle
from preprocessing import preprocessing
from model import LogReg
import time


app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Negative Speech Recognition',
    description='A Machine Learning API'
)


with open('model_logreg.pickle', 'rb') as f:
    model = pickle.load(f)

data_model = api.model('Text', {
  'text' : fields.String(required = True, description = 'text to predict')
})

def predict(inputs):
    prediction = model.predict(inputs)
    if prediction[0] == 1:
        pred_text = 'Negative'
    else:
        pred_text = 'Not Negative'
    return pred_text

@api.route('/label')
class Label(Resource):
    @api.expect(data_model)
    def post(self):
        start_time = time.time()
        inputs = api.payload
        inputs = preprocessing(inputs['text'])
        label = predict(inputs)
        end_time = time.time()
        duration = f'{end_time - start_time} seconds'
        response= {'data' : {'label': label, 'text' : api.payload['text']},
                    'description' : 'Text Classification Preicted',
                    'processing_time' : duration,
                    "status_code" : 200}
        return response, 200
    
    @api.route('/label/batch')
    class Label_batch(Resource):
        @api.expect([data_model])
        def post(self):
          start_time = time.time()
          inputs = api.payload
          batch = []
          for i in inputs:
            raw = i['text']
            data = preprocessing(raw)
            label = predict(data)
            batch.append({'label' : label, 'text' : raw})
          end_time = time.time()
          duration = f'{end_time - start_time} seconds'
          response= {'data' : batch,
                      'description' : 'Text Classification Preicted',
                      'processing_time' : duration,
                      "status_code" : 200}
          return response, 200

if __name__ == "__main__":
    app.run(debug=True)