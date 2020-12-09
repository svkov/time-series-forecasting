from flask_restful import Api

from api import DB, IsExist, Data, Hello
from app import app


def route():
    api.add_resource(Hello, '/')
    api.add_resource(Data, '/data')
    api.add_resource(DB, '/download')
    api.add_resource(IsExist, '/exist')


api = Api(app)
