from flask_restful import Api

from api import Hello
from app import app


def route():
    api.add_resource(Hello, '/')

api = Api(app)
