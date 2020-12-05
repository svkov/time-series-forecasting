from flask import Flask

from flask_sqlalchemy import SQLAlchemy
from db_config import DATABASE_CONNECTION_URI

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_CONNECTION_URI

db = SQLAlchemy(app)
from routes import route
db.create_all()
db.session.commit()


route()
