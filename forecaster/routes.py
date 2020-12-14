from flask_restful import Api

from api import Hello, DataLoaderAPI, AvailableModels
from app import app


def route():
    api.add_resource(Hello, '/')
    api.add_resource(AvailableModels, '/models')
    api.add_resource(DataLoaderAPI, '/<string:ticker>/<string:model>/')


api = Api(app)
