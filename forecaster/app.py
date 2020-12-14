from flask import Flask

from flask_sqlalchemy import SQLAlchemy
from config import DATABASE_CONNECTION_URI
from flask_cors import CORS

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_CONNECTION_URI
CORS(app)

db = SQLAlchemy(app)
from routes import route
db.create_all()
db.session.commit()


route()
