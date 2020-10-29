import os
import json

import pandas as pd
from flask import Flask
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
        input_data = pd.DataFrame()
        input_data = preprocess(input_data)
        prediction = predict(model, input_data).values[0]
        return prediction, 200


class Version(Resource):
    def get(self):
        with open("version.json", "r") as f:
            version = json.load(f)
        return version


API.add_resource(Version, '/')
API.add_resource(Predict, "/predict")


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=3333)
