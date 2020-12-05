from flask_restful import Api

from api import Hello, DataLoaderAPI
from app import app


def route():
    # api.add_resource(Hello, '/')
    api.add_resource(DataLoaderAPI, '/')

api = Api(app)
