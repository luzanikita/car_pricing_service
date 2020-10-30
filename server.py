import os
import json
import requests

import pandas as pd
from flask import Flask, render_template, make_response, request
from flask_restful import Api, Resource, reqparse

from regressor import predict, load_model
from preprocessing import preprocess


APP = Flask(__name__)
API = Api(APP)


class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument("engine_capacity", type=float)
        parser.add_argument("type", type=str)
        parser.add_argument("registration_year", type=int)
        parser.add_argument("gearbox", type=str)
        parser.add_argument("power", type=int)
        parser.add_argument("model", type=str)
        parser.add_argument("mileage", type=int)
        parser.add_argument("fuel", type=str)
        parser.add_argument("brand", type=str)
        parser.add_argument("damage", type=int)
        parser.add_argument("zipcode", type=int)
        parser.add_argument("insurance_price", type=float)

        args = parser.parse_args()
        model = load_model("model.pkl")
        input_data = pd.DataFrame([dict(args)])
        input_data = preprocess(input_data)
        print(input_data)
        prediction = round(predict(model, input_data)[0], 2)
        json_output = {
            "prediction": prediction
        }
        return json_output, 200



class Version(Resource):
    def get(self):
        with open("version.json", "r") as f:
            version = json.load(f)
        return version


class Main(Resource):
    def get(self):
        return make_response(
            render_template('index.html', message=""),
            200
        )
    
    def post(self):
        body = request.form.to_dict()
        r = requests.post(f'http://{request.host}/predict', data=body)
        prediction = eval(r.text)["prediction"]
        return make_response(
            render_template('index.html', message=prediction),
            200
        )


API.add_resource(Main, '/')
API.add_resource(Version, '/version')
API.add_resource(Predict, "/predict")


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=3333)
